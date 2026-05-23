# data/

Fine-tuning data for the UQ BIT information assistant. All files here are ready to use with Unsloth (Qwen2.5-14B, Gemma latest) or any HuggingFace-compatible trainer.

## Files

| File | Pairs | Description |
|------|-------|-------------|
| `train.jsonl` | 2,209 | Training set — AI-generated + human-validated pairs |
| `test.jsonl` | 50 | Evaluation set — **human-validated only** (corrected + manual) |
| `few_shot_examples.json` | 5 | Held-out human pairs for inference-time few-shot prompting |
| `split_stats.json` | — | Split metadata: seed, source counts, system prompt |
| `scripts/make_splits.py` | — | Script that produced all files above (re-runnable) |

---

## Source datasets

| Source | File | Pairs | Quality |
|--------|------|-------|---------|
| Human-authored | `data-collection/manual/data-manual.xlsx` | 38 | Highest |
| Human-corrected | `data-collection/generated/corrected/corrected-qa.csv` | 183 | High |
| AI-generated | `data-collection/generated/generated-qa-combined.csv` | 2,043 | Verified (RULES-grounded) |

The 221 human-validated pairs (38 + 183) are allocated as follows — test and few-shot are drawn exclusively from this pool:

```
221 human pairs
├─  5  → few_shot_examples.json   (inference only — excluded from train & test)
├─ 50  → test.jsonl               (evaluation)
└─ 166 → train.jsonl              (combined with generated)
```

All three sets are **fully disjoint** (seed 42, stratified by source).

---

## Format

Every line in `train.jsonl` and `test.jsonl` is a JSON object in HuggingFace messages format:

```json
{
  "messages": [
    {"role": "system",    "content": "You are a helpful information assistant for the UQ Bachelor of Information Technology program. Answer student questions accurately and concisely. Only provide information you are confident about. If you are not certain about a specific fact, say so and direct the student to study.uq.edu.au rather than guessing."},
    {"role": "user",      "content": "What is the minimum ATAR for the BIT?"},
    {"role": "assistant", "content": "The minimum entry score for the BIT (program 2570) is 81.9 for Semester 1, 2026 entry."}
  ]
}
```

`apply_chat_template` converts this to model-specific tokens (ChatML for Qwen, turn tokens for Gemma) automatically.

---

## Training with Unsloth (LoRA, A100)

```python
from datasets import load_dataset
from trl import SFTTrainer

ds = load_dataset("json", data_files={
    "train": "data/train.jsonl",
    "test":  "data/test.jsonl",
})

# Apply the model's chat template
def fmt(ex):
    return {"text": tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False)}

ds = ds.map(fmt, remove_columns=["messages"])

trainer = SFTTrainer(
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    dataset_text_field="text",
    # ... LoRA config, training args, etc.
)
trainer.train()
```

---

## Inference with few-shot prompting

Load `few_shot_examples.json` and prepend its `turns` after the system message:

```python
import json

fs = json.load(open("data/few_shot_examples.json"))

messages = [
    {"role": "system", "content": fs["system_prompt"]},
    *fs["turns"],                              # 5 example Q/A turns
    {"role": "user", "content": question},     # actual student question
]

inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
).to(model.device)

outputs = model.generate(inputs, max_new_tokens=256)
```

The few-shot examples were held out from both train and test — the model has never seen them during training, so they provide genuinely new context at inference time.

---

## Regenerating

```powershell
# From repo root
python data/scripts/make_splits.py
```

Deterministic — identical output for the same seed. Change `SEED`, `FEW_SHOT_SIZE`, or `TEST_SIZE` at the top of `make_splits.py` to adjust.
