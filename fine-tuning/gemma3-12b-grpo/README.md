# Gemma 3 12B — GRPO + QLoRA Fine-tuning

Reinforcement learning fine-tuning of `google/gemma-3-12b-it` for the UQ BIT information assistant, using:

- **Unsloth** `FastLanguageModel` — ~2× faster training, ~30% less VRAM vs vanilla HuggingFace
- **GRPO** (Group Relative Policy Optimization) via TRL `GRPOTrainer`
- **QLoRA** (4-bit NF4 + LoRA via Unsloth) for parameter-efficient training on a single A100 80 GB
- **G-Eval** reward via OpenAI GPT-4o-mini — scores completions on factual accuracy, relevance, conciseness, and no-hallucination

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters — edit here before running |
| `reward.py` | G-Eval reward function (OpenAI GPT scorer) |
| `train.py` | Main GRPO training script |

---

## Prerequisites

```bash
# Python 3.11+, CUDA 12.1+

# Install Unsloth first (match your CUDA + torch version)
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"

# Then the rest
pip install -r requirements.txt
```

### Environment variables (required)

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | HuggingFace token — Gemma 3 is a gated model, accept licence at hf.co/google/gemma-3-12b-it |
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
- Case 1 (accurate answer): ~1.0
- Case 2 (vague/safe answer): ~0.2–0.4
- Case 3 (hallucinated answer): ~0.2–0.4

### Step 2: Smoke test (5 GRPO steps)

Set `max_steps = 5` in `config.py`, then:

```bash
python fine-tuning/gemma3-12b-grpo/train.py
```

Confirms the full pipeline — model loads, LoRA applies, G=8 completions generate, G-Eval scores, one update step — without OOM or API errors.

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
| `per_device_train_batch_size` | 2 | Reduce to 1 if needed |
| `gradient_accumulation_steps` | 4 | Effective batch = batch × accum |
| `kl_coeff` | 0.1 | β — increase to 0.2 if KL diverges |
| `lora_r` | 64 | LoRA rank — 32 saves memory |
| `geval_model` | `gpt-4o-mini` | Switch to `gpt-4o` for higher quality scores |
| `max_steps` | -1 | Set to 5 for smoke test |

---

## Monitoring with Weights & Biases

Training logs to the `uq-unibot / uni-bot` project on wandb.ai. Key metrics:

| Metric | What to watch for |
|--------|------------------|
| `train/reward_mean` | Should trend upward over training |
| `train/kl` | Should stay < 5 (increase `kl_coeff` if it blows up) |
| `eval/reward_mean` | Primary generalisation signal |

---

## Loading the Trained Adapter for Inference

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="fine-tuning/gemma3-12b-grpo/checkpoints/final",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)  # enable Unsloth's optimised inference kernel

# Few-shot inference using held-out examples
import json
fs = json.load(open("data/few_shot_examples.json"))

messages = [
    {"role": "system", "content": fs["system_prompt"]},
    *fs["turns"],                        # 5 held-out Q&A examples
    {"role": "user", "content": "What is the minimum ATAR for the BIT?"},
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt",
                                       add_generation_prompt=True).to(model.device)
outputs = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

---

## Hardware Budget (single A100 80 GB)

| Component | Est. VRAM |
|-----------|-----------|
| Gemma 3 12B in NF4 4-bit | ~7 GB |
| LoRA adapters (r=64) | ~0.5 GB |
| KV cache — 8 completions × 256 tokens × batch 2 | ~14 GB |
| Optimizer states (LoRA params only) | ~2 GB |
| Activations + overhead | ~5 GB |
| **Total** | **~29 GB** — safe headroom in 80 GB |

---

## Cost Estimate

| Resource | Estimate |
|----------|---------|
| G-Eval (GPT-4o-mini) | ~$5–10 USD per full training run (1000 steps, G=8) |
