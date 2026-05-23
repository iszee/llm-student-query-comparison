"""
train.py
--------
GRPO + QLoRA fine-tuning of Gemma 3 12B for the UQ BIT information assistant.

Algorithm: Group Relative Policy Optimization (GRPO) via TRL GRPOTrainer.
  - Samples G=8 completions per prompt
  - Scores each with G-Eval (OpenAI GPT-4o-mini)
  - Updates LoRA adapters using relative reward advantage

Quantisation: 4-bit NF4 (bitsandbytes) + LoRA (PEFT).
Optimiser:    Paged AdamW 8-bit (bitsandbytes) — ~30% less optimizer VRAM.
Generation:   vLLM (PagedAttention) — ~10-20× faster generation than HF generate().

NOTE: Unsloth was removed due to a bug in Unsloth 2025.11.1 (VARIANT_KWARG_KEYS
undefined in compiled Linear_peft_forward.py). Vanilla HuggingFace PEFT + TRL
works correctly once the CUDA 13.0 library path is set:
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<venv>/site-packages/nvidia/cu13/lib/

Flash Attention 2 requires:
  pip install flash-attn --no-build-isolation

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

import huggingface_hub
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import GRPOConfig, GRPOTrainer

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


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: Config):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model: {cfg.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,               # renamed from torch_dtype (deprecated)
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # suppresses requires_grad warning
    )

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
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

    os.environ["WANDB_ENTITY"]  = cfg.wandb_entity
    os.environ["WANDB_PROJECT"] = cfg.wandb_project

    # Log in to HuggingFace Hub (required for gated Gemma 3 weights)
    huggingface_hub.login(token=os.environ["HF_TOKEN"])

    # ── Load dataset ───────────────────────────────────────────────────────────
    print("Loading dataset...")
    ds = load_dataset("json", data_files={
        "train": cfg.train_file,
        "test":  cfg.test_file,
    })

    # ── Load model + tokenizer ─────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg)

    # ── Prepare dataset columns ────────────────────────────────────────────────
    prep = partial(prepare_example, tokenizer=tokenizer)
    ds = ds.map(prep, remove_columns=["messages"])

    # ── GRPOConfig ─────────────────────────────────────────────────────────────
    grpo_cfg = GRPOConfig(
        # Generation — vLLM handles forward pass; HF model does backward
        use_vllm=cfg.use_vllm,
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        vllm_dtype=cfg.vllm_dtype,
        vllm_max_model_len=cfg.vllm_max_model_len,
        num_generations=cfg.num_generations,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        # KL penalty
        beta=cfg.beta,
        # Optimiser
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # suppresses requires_grad warning
        optim="paged_adamw_8bit",           # 8-bit paged optimizer — saves ~30% optimizer VRAM
        top_p=0.95,                         # match Gemma 3 model default (suppresses generation_config warning)
        # Logging / saving
        output_dir=cfg.output_dir,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to,
        run_name=cfg.run_name,
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
