"""
train.py
--------
GRPO + LoRA fine-tuning of Mistral Nemo 12B for the UQ BIT information assistant.

Algorithm: Group Relative Policy Optimization (GRPO) via TRL GRPOTrainer.
  - Samples G=4 completions per prompt
  - Scores each with G-Eval (OpenAI GPT-4o-mini)
  - Updates LoRA adapters using relative reward advantage

Precision:  BF16 full-precision base model + LoRA adapters via PEFT (dtype=torch.bfloat16).
Optimiser:  AdamW fused (torch) — fast, no bitsandbytes dependency.
Generation: vLLM colocate mode (PagedAttention) — ~10-20× faster than HF generate().

PEFT integration uses TRL v1.4+ native peft_config parameter — pass LoraConfig
directly to GRPOTrainer; the trainer wraps the model and enables gradient
checkpointing internally.

Attention backend (set in load_model_and_tokenizer):
  "sdpa"              — default; PyTorch built-in SDPA, no extra install (torch >= 2.0)
  "flash_attention_2" — fastest; requires: pip install flash-attn --no-build-isolation

Usage:
    python fine-tuning/mistral-nemo-12b-grpo/train.py

For a 5-step smoke test, set max_steps=5 in config.py before running.

Requires env vars:
    OPENAI_API_KEY   — for G-Eval scoring
    WANDB_API_KEY    — for Weights & Biases logging
    HF_TOKEN         — for gated Mistral Nemo weights on HuggingFace Hub
"""

import os
import sys
import torch
from pathlib import Path
from functools import partial

import huggingface_hub
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    """Load base model and tokenizer in BF16. PEFT wrapping is handled by GRPOTrainer."""
    print(f"Loading model: {cfg.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",    # PyTorch built-in SDPA — faster than "eager", no extra install
        # attn_implementation="flash_attention_2",  # fastest; requires: pip install flash-attn --no-build-isolation
    )

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

    # ── vLLM / Triton cache ────────────────────────────────────────────────────
    # Set before any vLLM import resolves its cache paths.  On HPC systems the
    # default ~/.cache/vllm and ~/.triton/cache may not be writable; redirect
    # both to a project-local directory the job always owns.
    vllm_cache = Path(cfg.vllm_cache_dir).resolve()
    vllm_cache.mkdir(parents=True, exist_ok=True)
    (vllm_cache / "triton").mkdir(exist_ok=True)
    os.environ.setdefault("VLLM_CACHE_ROOT",  str(vllm_cache))
    os.environ.setdefault("TRITON_CACHE_DIR", str(vllm_cache / "triton"))

    # ── Validate env vars ──────────────────────────────────────────────────────
    missing = [v for v in ("OPENAI_API_KEY", "WANDB_API_KEY", "HF_TOKEN")
               if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    os.environ["WANDB_ENTITY"]  = cfg.wandb_entity
    os.environ["WANDB_PROJECT"] = cfg.wandb_project

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

    # ── LoRA config (passed to GRPOTrainer — trainer handles PEFT wrapping) ────
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── GRPOConfig ─────────────────────────────────────────────────────────────
    grpo_cfg = GRPOConfig(
        # Generation — vLLM colocate mode shares GPU memory with training model
        use_vllm=cfg.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        vllm_max_model_length=cfg.vllm_max_model_length,
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
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=cfg.optim,
        # Logging / saving / evaluation
        output_dir=cfg.output_dir,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps",
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
        peft_config=lora_cfg,
    )

    # Print trainable parameter count after trainer wraps the model with PEFT
    trainer.model.print_trainable_parameters()

    print("Starting GRPO training...")
    trainer.train()

    # ── Save final adapter ─────────────────────────────────────────────────────
    final_dir = Path(cfg.output_dir) / "final"
    print(f"Saving adapter to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    print("Done.")


if __name__ == "__main__":
    main()
