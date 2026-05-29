# Qwen3 14B — GRPO Fine-tuning

Reinforcement learning fine-tuning of `Qwen/Qwen3-14B` for the UQ BIT information assistant, using:

- **GRPO** (Group Relative Policy Optimization) via TRL `GRPOTrainer`
- **BF16 + LoRA** (full-precision BF16 base + LoRA via PEFT) — parameter-efficient training on a single H100 79 GB
- **vLLM** (PagedAttention, colocate mode) for generation — ~10–20× faster than HuggingFace `generate()`
- **G-Eval** reward via OpenAI GPT-4o-mini — scores completions on factual accuracy, relevance, conciseness, and no-hallucination

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters — edit here before running |
| `reward.py` | G-Eval reward function (OpenAI GPT-4o-mini scorer) |
| `train.py` | Main GRPO training script |
| `evaluate.py` | Evaluate a checkpoint across all 8 prompt configurations |

---

## Prerequisites

```bash
# Python 3.11+, CUDA 12.1+
pip install -r requirements.txt
```

### Environment variables (required)

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | HuggingFace token — Qwen3-14B is an open model; needed for Hub API access |
| `OPENAI_API_KEY` | OpenAI API key for G-Eval scoring (GPT-4o-mini) |
| `WANDB_API_KEY` | Weights & Biases API key for experiment tracking |

```bash
export HF_TOKEN="..."
export OPENAI_API_KEY="..."
export WANDB_API_KEY="..."
```

---

## Running

### Step 1: Test G-Eval connectivity (no GPU needed)

```bash
python fine-tuning/qwen3-14B-grpo/reward.py
```

Expected output — three cases with clearly separated rewards:
- Case 1 (accurate answer): ~0.9–1.0
- Case 2 (vague/safe answer): ~0.2–0.4
- Case 3 (hallucinated answer): ~0.1–0.3

### Step 2: Smoke test (5 GRPO steps)

Set `max_steps = 5` in `config.py`, then:

```bash
python fine-tuning/qwen3-14B-grpo/train.py
```

### Step 3: Full training run

Reset `max_steps = -1` in `config.py`, then:

```bash
python fine-tuning/qwen3-14B-grpo/train.py
```

### Step 4: Evaluate (all 8 configurations by default)

```bash
# Full 8-config run: base + fine-tuned × 4 prompt variants
python fine-tuning/qwen3-14B-grpo/evaluate.py

# Skip fine-tuned model — base model only (4 configs)
python fine-tuning/qwen3-14B-grpo/evaluate.py --checkpoint none

# Specific mid-training checkpoint
python fine-tuning/qwen3-14B-grpo/evaluate.py \
    --checkpoint fine-tuning/qwen3-14B-grpo/checkpoints/checkpoint-50

# Dry run — generation only, no G-Eval cost
python fine-tuning/qwen3-14B-grpo/evaluate.py --no-geval
```

Saves to `fine-tuning/qwen3-14B-grpo/results/`:

| File | Contents |
|------|----------|
| `eval_base_plain.csv` | Base model, no system prompt, no few-shot |
| `eval_base_sysprompt.csv` | Base model + system prompt |
| `eval_base_fewshot.csv` | Base model + few-shot examples |
| `eval_base_sysprompt_fewshot.csv` | Base model + system prompt + few-shot |
| `eval_finetuned_plain.csv` | Fine-tuned model, no system prompt, no few-shot |
| `eval_finetuned_sysprompt.csv` | Fine-tuned model + system prompt |
| `eval_finetuned_fewshot.csv` | Fine-tuned model + few-shot examples |
| `eval_finetuned_sysprompt_fewshot.csv` | Fine-tuned model + system prompt + few-shot |
| `eval_summary.csv` | **One row per config — mean metrics across all 8** |

---

## Configuration

Edit `config.py` to adjust hyperparameters. Key knobs:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_generations` | 4 | G completions per prompt — increase to 8 for stronger advantage signal (more VRAM) |
| `per_device_train_batch_size` | 4 | Reduce to 2 if OOM |
| `gradient_accumulation_steps` | 4 | Effective batch = 4 × 4 = 16 prompts |
| `max_completion_length` | 256 | Max tokens per completion |
| `temperature` | 0.8 | Sampling temp for G completions |
| `learning_rate` | 5e-6 | Conservative for GRPO stability |
| `beta` | 0.1 | KL penalty — set to 0.0 to disable |
| `optim` | `adamw_torch_fused` | Fused AdamW (torch); no bitsandbytes |
| `lora_r` | 64 | LoRA rank — 32 saves memory |
| `use_vllm` | `True` | Disable if vLLM install fails (slower but works) |
| `vllm_gpu_memory_utilization` | 0.45 | ~35.5 GB on H100 79 GB (higher than Gemma 12B due to larger model) |
| `vllm_cache_dir` | `fine-tuning/qwen3-14B-grpo/cache/vllm` | Writable cache for HPC systems |
| `geval_model` | `gpt-4o-mini` | Switch to `gpt-4o` for higher quality scores |
| `max_steps` | -1 | Set to 5 for smoke test |

---

## Hardware Budget (H100 PCIe 79 GB)

| Component | Est. VRAM |
|-----------|-----------|
| vLLM — Qwen3 14B BF16 (colocate, KV cache at 0.45 util) | ~35.5 GB |
| Training model — Qwen3 14B BF16 (full precision) | ~28 GB |
| LoRA adapters (r=64) | ~0.5 GB |
| AdamW fused optimizer states | ~3 GB |
| Activations + overhead (gradient checkpointing ON) | ~3 GB |
| **Total** | **~70 GB** — safe headroom in 79 GB |

---

## Cost Estimate

| Resource | Estimate |
|----------|---------|
| G-Eval (GPT-4o-mini) | ~$2–5 USD per full training run |
| Bunya H100 compute | ~6–10 hours (14B is ~25% slower per step than 12B) |
