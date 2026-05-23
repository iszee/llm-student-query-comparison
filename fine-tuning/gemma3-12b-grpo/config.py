"""
config.py
---------
Central configuration for Gemma 3 12B GRPO + QLoRA fine-tuning.
All hyperparameters live here — import Config from this module in train.py and reward.py.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Model ─────────────────────────────────────────────────────────────────
    model_id: str = "google/gemma-3-12b-it"
    load_in_4bit: bool = True           # NF4 QLoRA via bitsandbytes

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    gradient_checkpointing: bool = True

    # ── GRPO ──────────────────────────────────────────────────────────────────
    num_generations: int = 4            # G completions sampled per prompt (was 8, halved for speed)
    max_completion_length: int = 128    # max tokens per completion (was 256, halved for speed)
    temperature: float = 0.9            # sampling temperature for diverse completions
    beta: float = 0.1                   # KL penalty weight (was kl_coeff in TRL <0.15)

    # ── Training ──────────────────────────────────────────────────────────────
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 4    # was 2; larger batch → fewer steps
    gradient_accumulation_steps: int = 4    # effective batch = 16
    num_train_epochs: int = 1               # was 3; 1 epoch sufficient for GRPO
    max_steps: int = -1                     # set to small number (e.g. 5) for smoke test
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    output_dir: str = "fine-tuning/gemma3-12b-grpo/checkpoints"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3

    # ── Data ──────────────────────────────────────────────────────────────────
    train_file: str = "data/train.jsonl"
    test_file: str = "data/test.jsonl"
    max_prompt_length: int = 512        # truncate formatted prompt if needed

    # ── G-Eval (OpenAI) ───────────────────────────────────────────────────────
    geval_model: str = "gpt-4o-mini"    # cheap + capable enough for scoring
    geval_max_retries: int = 3
    geval_timeout: float = 30.0
    geval_max_workers: int = 16         # parallel threads for OpenAI API calls
    # Dimension weights — must sum to 1.0
    geval_weights: dict = field(default_factory=lambda: {
        "factual_accuracy": 0.55,
        "relevance":        0.25,
        "conciseness":      0.10,
        "no_hallucination": 0.10,
    })

    # ── Weights & Biases ──────────────────────────────────────────────────────
    wandb_entity: str = "uq-unibot"     # W&B team/org (set via WANDB_ENTITY env var)
    wandb_project: str = "uni-bot"      # W&B project name (no slashes)
    run_name: str = "gemma3-12b-grpo-geval"
    report_to: str = "wandb"            # set to "none" to disable W&B
