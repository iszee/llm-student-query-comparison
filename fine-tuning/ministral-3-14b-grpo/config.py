"""
config.py
---------
Central configuration for Ministral 3 14B GRPO + LoRA (BF16) fine-tuning.
All hyperparameters live here — import Config from this module in train.py and reward.py.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Model ─────────────────────────────────────────────────────────────────
    model_id: str = "mistralai/Ministral-3-14B-Instruct-2512-BF16"

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
    num_generations: int = 4             # G completions sampled per prompt
    max_completion_length: int = 256     # max tokens per completion (eval avg ~150 tokens)
    temperature: float = 0.2            # sampling temperature — low for factual consistency
    beta: float = 0.1                   # KL penalty weight (set to 0.0 to disable)

    # ── Training ──────────────────────────────────────────────────────────────
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 4    # 4 prompts × 4 generations = 16 seqs per step
    gradient_accumulation_steps: int = 4    # effective batch = 4×4 = 16 prompts
    num_train_epochs: int = 1               # 1 epoch sufficient for GRPO
    max_steps: int = -1                     # set to small number (e.g. 5) for smoke test
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    optim: str = "adamw_torch_fused"        # fused AdamW — fast on CUDA, no bitsandbytes
    output_dir: str = "fine-tuning/ministral-3-14b-grpo/checkpoints"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 10

    # ── Data ──────────────────────────────────────────────────────────────────
    train_file: str = "data/train.jsonl"
    test_file: str = "data/test.jsonl"
    max_prompt_length: int = 1024       # tokenizer truncation limit used in evaluate.py (not passed to GRPOConfig)

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

    # ── vLLM (fast generation) ────────────────────────────────────────────────
    # vLLM runs in colocate mode: shares the training GPU, syncs LoRA weights after
    # each optimizer step. TRL handles the weight sync automatically.
    use_vllm: bool = True
    vllm_gpu_memory_utilization: float = 0.45   # ~35.5 GB on H100 79 GB; 14B needs more than 12B
    vllm_max_model_length: int = 2048           # prompt(1024) + completion(256) + safety margin
    vllm_cache_dir: str = "fine-tuning/ministral-3-14b-grpo/cache/vllm"  # writable cache (VLLM_CACHE_ROOT + TRITON_CACHE_DIR)

    # ── Weights & Biases ──────────────────────────────────────────────────────
    wandb_entity: str = "uq-unibot"
    wandb_project: str = "uni-bot"
    run_name: str = "ministral-3-14b-grpo-geval"
    report_to: str = "wandb"
