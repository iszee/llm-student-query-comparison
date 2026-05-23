"""
train.py
--------
GRPO + QLoRA fine-tuning of Gemma 3 12B using Unsloth for the UQ BIT
information assistant.

Algorithm: Group Relative Policy Optimization (GRPO) via TRL GRPOTrainer.
  - Samples G=8 completions per prompt
  - Scores each with G-Eval (OpenAI GPT-4o-mini)
  - Updates LoRA adapters using relative reward advantage

Unsloth replaces the manual BitsAndBytesConfig + PEFT setup with
FastLanguageModel, giving ~2x faster training and ~30% less VRAM vs vanilla
HuggingFace on A100.

Usage:
    python fine-tuning/gemma3-12b-grpo/train.py

For a 5-step smoke test, set max_steps=5 in config.py before running.

Requires env vars:
    OPENAI_API_KEY   — for G-Eval scoring
    WANDB_API_KEY    — for Weights & Biases logging
    HF_TOKEN         — for gated Gemma 3 weights on HuggingFace Hub
"""

import os
import sys
import torch
from pathlib import Path
from functools import partial

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from reward import reward_fn

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful information assistant for the UQ Bachelor of Information "
    "Technology program. Answer student questions accurately and concisely. "
    "Only provide information you are confident about. "
    "If you are not certain about a specific fact, say so and direct the student "
    "to study.uq.edu.au rather than guessing."
)


# ── Dataset preparation ───────────────────────────────────────────────────────

def prepare_example(example: dict, tokenizer) -> dict:
    """
    Convert a HF-messages-format example into:
      - "prompt":    formatted string (system + user turn, with generation prompt)
      - "question":  raw question string (passed through to G-Eval via kwargs)
      - "reference": reference answer string (passed through to G-Eval via kwargs)
    """
    msgs = example["messages"]
    question = msgs[1]["content"]   # user turn
    answer   = msgs[2]["content"]   # assistant turn

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return {"prompt": prompt, "question": question, "reference": answer}


# ── Model loading (Unsloth) ───────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: Config):
    dtype = getattr(torch, cfg.dtype) if cfg.dtype else None

    print(f"Loading model with Unsloth: {cfg.model_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_id,
        max_seq_length=cfg.max_seq_length,
        dtype=dtype,
        load_in_4bit=cfg.load_in_4bit,
    )

    # Apply LoRA via Unsloth (fused kernels, optimised checkpointing)
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        random_state=42,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"     # GRPO needs left-padding for generation

    return model, tokenizer


# ── Reward wrapper ────────────────────────────────────────────────────────────

def make_reward_fn(cfg: Config):
    """Wrap reward_fn with config so GRPOTrainer can call it as a plain function."""
    def _reward(prompts, completions, **kwargs):
        return reward_fn(prompts, completions, cfg=cfg, **kwargs)
    return _reward


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = Config()

    # ── Validate env vars ──────────────────────────────────────────────────────
    missing = [v for v in ("OPENAI_API_KEY", "WANDB_API_KEY", "HF_TOKEN")
               if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    os.environ["WANDB_PROJECT"] = cfg.wandb_project

    # ── Load dataset ───────────────────────────────────────────────────────────
    print("Loading dataset...")
    ds = load_dataset("json", data_files={
        "train": cfg.train_file,
        "test":  cfg.test_file,
    })

    # ── Load model + tokenizer via Unsloth ────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg)

    # ── Prepare dataset columns ────────────────────────────────────────────────
    prep = partial(prepare_example, tokenizer=tokenizer)
    ds = ds.map(prep, remove_columns=["messages"])

    # ── GRPOConfig ─────────────────────────────────────────────────────────────
    grpo_cfg = GRPOConfig(
        # Generation
        num_generations=cfg.num_generations,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        # KL
        kl_coeff=cfg.kl_coeff,
        # Optimiser
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        bf16=cfg.bf16,
        # Logging / saving
        output_dir=cfg.output_dir,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to,
        run_name=cfg.run_name,
        # Pass dataset columns through to reward_fn as kwargs
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # ── GRPOTrainer ────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_reward_fn(cfg),
        args=grpo_cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    trainer.train()

    # ── Save final adapter ─────────────────────────────────────────────────────
    final_dir = Path(cfg.output_dir) / "final"
    print(f"Saving adapter to {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Done.")


if __name__ == "__main__":
    main()
