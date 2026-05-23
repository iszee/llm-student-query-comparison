# Gemma 3 12B — GRPO + QLoRA Fine-tuning

Reinforcement learning fine-tuning of `google/gemma-3-12b-it` for the UQ BIT information assistant, using:

- **GRPO** (Group Relative Policy Optimization) via TRL `GRPOTrainer`
- **QLoRA** (4-bit NF4 via bitsandbytes + LoRA via PEFT) — parameter-efficient training on a single H100 79 GB
- **vLLM** (PagedAttention) for generation — ~10–20× faster than HuggingFace `generate()`
- **G-Eval** reward via OpenAI GPT-4o-mini — scores completions on factual accuracy, relevance, conciseness, and no-hallucination

> **Note on Unsloth:** Removed due to a bug in Unsloth 2025.11.1 (`VARIANT_KWARG_KEYS` undefined in compiled `Linear_peft_forward.py`). Vanilla HuggingFace PEFT + TRL + vLLM achieves comparable or better throughput.

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters — edit here before running |
| `reward.py` | G-Eval reward function (OpenAI GPT-4o-mini scorer) |
| `train.py` | Main GRPO training script |

---

## Prerequisites

```bash
# Python 3.11+, CUDA 12.1+

# Set the CUDA 13.0 library path (required for bitsandbytes on Bunya)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<venv>/lib/python3.11/site-packages/nvidia/cu13/lib/

# Install vLLM
pip install vllm

# Install remaining dependencies
pip install -r requirements.txt
```

### Environment variables (required)

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | HuggingFace token — Gemma 3 is a gated model; accept licence at hf.co/google/gemma-3-12b-it |
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
python fine-tuning/gemma3-12b-grpo/reward.py
```

Expected output — three cases with clearly separated rewards:
- Case 1 (accurate answer): ~0.9–1.0
- Case 2 (vague/safe answer): ~0.2–0.4
- Case 3 (hallucinated answer): ~0.1–0.3

### Step 2: Smoke test (5 GRPO steps)

Set `max_steps = 5` in `config.py`, then:

```bash
python fine-tuning/gemma3-12b-grpo/train.py
```

Confirms the full pipeline — model loads, vLLM starts, G=8 completions generate, G-Eval scores, one update step — without OOM or API errors.

### Step 3: Full training run

Reset `max_steps = -1` in `config.py`, then:

```bash
python fine-tuning/gemma3-12b-grpo/train.py
```

---

## Configuration

Edit `config.py` to adjust hyperparameters. Key knobs:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_generations` | 8 | G completions per prompt — reduce to 4 if OOM |
| `per_device_train_batch_size` | 8 | Reduce to 4 if OOM |
| `gradient_accumulation_steps` | 2 | Effective batch = 8 × 2 = 16 prompts |
| `max_completion_length` | 1024 | Max tokens per completion |
| `learning_rate` | 5e-6 | Conservative for GRPO stability |
| `beta` | 0.1 | KL penalty — increase to 0.2 if KL diverges |
| `lora_r` | 64 | LoRA rank — 32 saves memory |
| `use_vllm` | `True` | Disable if vLLM install fails (slower but works) |
| `vllm_gpu_memory_utilization` | 0.45 | ~35 GB on H100 79 GB; lower if OOM |
| `geval_model` | `gpt-4o-mini` | Switch to `gpt-4o` for higher quality scores |
| `max_steps` | -1 | Set to 5 for smoke test |

---

## How vLLM Integrates with GRPO

```
Each optimizer step:
  1. Generation  → vLLM (Gemma 12B BF16, PagedAttention) — 10-20× faster
  2. G-Eval      → 16 parallel OpenAI API calls per mini-batch
  3. Backward    → HF model (NF4 4-bit + LoRA, Flash Attention 2)
  4. Weight sync → TRL merges LoRA → pushes updated weights to vLLM
```

---

## Monitoring with Weights & Biases

Training logs to the `uq-unibot / uni-bot` project on wandb.ai. Key metrics:

| Metric | What to watch for |
|--------|------------------|
| `train/reward_mean` | Should trend upward over training |
| `train/reward_std` | Should be > 0.1 — low std means poor advantage signal |
| `train/kl` | Should stay < 5 — increase `beta` in `config.py` if it blows up |
| `completions/clipped_ratio` | Should be < 0.2 — high means `max_completion_length` is too short |
| `eval/reward_mean` | Primary generalisation signal |

---

## Loading the Trained Adapter for Inference

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

model_id = "google/gemma-3-12b-it"
adapter_path = "fine-tuning/gemma3-12b-grpo/checkpoints/final"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# Few-shot inference using held-out examples
fs = json.load(open("data/few_shot_examples.json"))

messages = [
    {"role": "system", "content": fs["system_prompt"]},
    *fs["turns"],                        # 5 held-out Q&A examples
    {"role": "user", "content": "What is the minimum ATAR for the BIT?"},
]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
).to(model.device)
outputs = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

---

## Hardware Budget (H100 PCIe 79 GB)

| Component | Est. VRAM |
|-----------|-----------|
| vLLM — Gemma 12B BF16 (generation engine) | ~24 GB |
| vLLM — KV cache (at 0.45 utilisation) | ~11 GB |
| Training model — Gemma 12B NF4 4-bit | ~7 GB |
| LoRA adapters (r=64) | ~0.5 GB |
| Paged AdamW 8-bit optimizer states | ~1 GB |
| Activations + overhead (gradient checkpointing ON) | ~3 GB |
| **Total** | **~47 GB** — safe headroom in 79 GB |

---

## Cost Estimate

| Resource | Estimate |
|----------|---------|
| G-Eval (GPT-4o-mini) | ~$2–5 USD per full training run (277 steps × 16 calls/step) |
| Bunya H100 compute | ~5–8 hours at estimated ~65 s/step |
