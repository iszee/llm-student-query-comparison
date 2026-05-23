# LLM Student Query Comparison

Research project at the University of Queensland comparing how large language models respond to student queries about UQ programs. The focus is benchmarking and evaluating LLM responses against official UQ program information and student guidance documents.

## Project Overview

The project targets the **Bachelor of Information Technology (BIT)** program at UQ (program codes 2570 / 2453) and its associated dual degrees. It builds datasets of student Q&A pairs — both human-authored and synthetic — to serve as evaluation benchmarks and fine-tuning data for information-assistant models.

## Environment Setup

```powershell
# Activate the Python virtual environment (Windows)
.\env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Repository Structure

```
llm-student-query-comparison/
├── data/                                             # Fine-tuning splits (see data/README.md)
│   ├── train.jsonl                                   # 2209 pairs (HF messages format)
│   ├── test.jsonl                                    # 50 pairs  (human-validated only)
│   ├── few_shot_examples.json                        # 5 held-out pairs for inference-time few-shot
│   ├── split_stats.json                              # Seed, source counts, split metadata
│   └── scripts/
│       └── make_splits.py                            # Reproducible split script (seed=42)
├── data-collection/
│   ├── manual/
│   │   └── data-manual.xlsx                          # 38 human-authored baseline Q&A pairs
│   ├── generated/
│   │   ├── generate_qa.py                            # v1 script (hardcoded, 200 pairs)
│   │   ├── generated-qa-200.xlsx                     # v1 output (200 AI-generated pairs)
│   │   ├── generate_qa_v2.py                         # v2 script (Claude API, 2000 pairs)
│   │   ├── generated-qa-2000.csv                     # v2 raw output (1799 unique pairs)
│   │   ├── generate_qa_extra.py                      # clean + top-up + dedup script
│   │   ├── generated-qa-extra-300.csv                # 244 additional unique pairs
│   │   ├── generated-qa-combined.csv                 # final dataset (2043 pairs, use this)
│   │   └── corrected/
│   │       └── corrected-qa.csv                      # 183 human-corrected pairs (ground truth)
│   └── sources/
│       ├── program-pages.txt                         # URLs to UQ BIT program pages
│       ├── redit-pages.txt                           # Reddit/forum discussion URLs (placeholder)
│       └── international-guide-undergraduate-postgraduate.pdf  # UQ International Guide 2026
├── fine-tuning/
│   └── gemma3-12b-grpo/                              # Gemma 3 12B GRPO+QLoRA (see README inside)
│       ├── config.py                                 # All hyperparameters
│       ├── reward.py                                 # G-Eval reward (OpenAI GPT-4o-mini)
│       └── train.py                                  # Unsloth + TRL GRPOTrainer
└── requirements.txt
```

## Datasets

### `data-manual.xlsx` — Human-authored baseline (38 pairs)
Manually curated Q&A pairs covering core BIT program topics. Used as the human baseline for comparison with LLM-generated responses.

### `generated-qa-200.xlsx` — First synthetic dataset (200 pairs)
AI-generated Q&A pairs produced by Claude (v1 script). This dataset contained factual errors and was subsequently corrected by hand.

### `corrected-qa.csv` — Human-corrected ground truth (183 pairs)
The 200 v1 pairs after manual review: 183 were kept (with corrections), 17 were removed. Key corrections included:
- Wrong ATAR scores (BBusMgt/IT: 84 → 86.4; BCom/IT: 84 → 84.4)
- Wrong application dates ("30 November" → "23 Feb 2026" / "27 Jul 2026")
- Wrong policy (BIT/Arts was stated to include an IT major — corrected to no IT major)
- Wrong dual-degree major policy (from 2026: only 1 major per program)
- Wrong fees (SSAF $365 → $373; living costs updated)
- Verbose answers condensed to 1–2 sentences

This file is the **validated ground truth** used as few-shot examples by the v2 generation script.

### `generated-qa-combined.csv` — Final fine-tuning dataset (2043 pairs) ✅
The primary dataset for fine-tuning. Produced by `generate_qa_extra.py` in three steps:
1. **Clean** — em dashes removed from the 1799 v2 pairs (replaced with hyphens)
2. **Top-up** — 300 new pairs generated (25 per topic), deduplicated against existing; 244 unique pairs retained
3. **Combine** — 1799 cleaned + 244 new = 2043 unique pairs after full deduplication

Covers 12 topic categories:

| Topic | v2 pairs | Extra pairs | Total |
|-------|----------|-------------|-------|
| Domestic Admission & Entry Requirements | 150 | 21 | 171 |
| International Admission & Entry Requirements | 200 | 16 | 216 |
| Program Structure & Duration | 150 | 23 | 173 |
| Majors & Specialisations | 150 | 21 | 171 |
| Specific Courses & Study Plans | 200 | 19 | 219 |
| Dual Degree Programs | 200 | 18 | 218 |
| Fees & Financial Matters | 150 | 21 | 171 |
| Campus Life & Student Support | 150 | 21 | 171 |
| Career Outcomes & Pathways | 150 | 23 | 173 |
| Honours Program | 150 | 24 | 174 |
| Application Process | 150 | 17 | 167 |
| Additional Topics | 200 | 20 | 220 |
| **Unique total** | **1799** | **244** | **2043** |

### `generated-qa-2000.csv` — v2 raw output (1799 unique pairs)
Intermediate file; superseded by `generated-qa-combined.csv`.

## Regenerating the Datasets

### Final combined dataset (recommended)

Requires an Anthropic API key. Runs clean + generate-300 + dedup in one pass:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
.\env\Scripts\Activate.ps1
python data-collection/generated/generate_qa_extra.py
```

- Resumes automatically if interrupted
- Estimated cost: ~$0.50–1 USD (12 batches of 25 pairs)

### v2 base dataset only

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python data-collection/generated/generate_qa_v2.py
```

- Generates 2000 pairs across 104 batches; resumes on interrupt
- Estimated cost: ~$3–6 USD with prompt caching

### v1 dataset (original, no API needed)

```powershell
python data-collection/generated/generate_qa.py
```

## Fine-tuning

Prepared splits live in `data/` — see [`data/README.md`](data/README.md) for format and usage.

| Model | Method | Directory |
|-------|--------|-----------|
| Gemma 3 12B | GRPO + QLoRA (Unsloth) + G-Eval reward | `fine-tuning/gemma3-12b-grpo/` |

G-Eval scores completions via **OpenAI GPT-4o-mini** on four dimensions (factual accuracy 55%, relevance 25%, conciseness 10%, no-hallucination 10%). W&B project: `uq-unibot/uni-bot`.

See [`fine-tuning/gemma3-12b-grpo/README.md`](fine-tuning/gemma3-12b-grpo/README.md) for setup and running instructions.

---

## Source Documents

| File | Description |
|------|-------------|
| `sources/program-pages.txt` | Official UQ program page URLs for BIT and 7 dual degrees |
| `sources/international-guide-undergraduate-postgraduate.pdf` | UQ International Student Guide 2026 — primary factual reference |
