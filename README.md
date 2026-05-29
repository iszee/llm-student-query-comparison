# LLM Student Query Comparison

Research project at the University of Queensland comparing how large language models respond to student queries about UQ programs. The focus is benchmarking and evaluating LLM responses against official UQ program information and student guidance documents.

## Project Overview

The project targets the **Bachelor of Information Technology (BIT)** program at UQ (program codes 2570 / 2453) and its associated dual degrees. It builds datasets of student Q&A pairs - both human-authored and synthetic - to serve as evaluation benchmarks and fine-tuning data for information-assistant models.

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
│   ├── gemma3-12b-grpo/                              # Gemma 3 12B GRPO + LoRA (see README inside)
│   │   ├── config.py                                 # All hyperparameters
│   │   ├── reward.py                                 # G-Eval reward (OpenAI GPT-4o-mini)
│   │   ├── train.py                                  # TRL GRPOTrainer (BF16 + LoRA + vLLM)
│   │   └── evaluate.py                               # 16-config ablation evaluator (8 + 8 RAG)
│   ├── mistral-nemo-12b-grpo/                         # Mistral Nemo 12B GRPO + LoRA (see README inside)
│   │   ├── config.py
│   │   ├── reward.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── qwen3-14B-grpo/                               # Qwen3 14B GRPO + LoRA (see README inside)
│       ├── config.py
│       ├── reward.py
│       ├── train.py
│       └── evaluate.py
├── rag/                                              # Retrieval-Augmented Generation module
│   ├── __init__.py
│   ├── __main__.py                                   # CLI: python -m rag build | query "..."
│   ├── build_index.py                                # PDF + URL ingest → chunk → contextualise → embed → index
│   └── retrieve.py                                   # Hybrid retriever: dense + BM25 + RRF + rerank
└── requirements.txt
```

## Datasets

### `data-manual.xlsx` - Human-authored baseline (38 pairs)
Manually curated Q&A pairs covering core BIT program topics. Used as the human baseline for comparison with LLM-generated responses.

### `generated-qa-200.xlsx` - First synthetic dataset (200 pairs)
AI-generated Q&A pairs produced by Claude (v1 script). This dataset contained factual errors and was subsequently corrected by hand.

### `corrected-qa.csv` - Human-corrected ground truth (183 pairs)
The 200 v1 pairs after manual review: 183 were kept (with corrections), 17 were removed. Key corrections included:
- Wrong ATAR scores (BBusMgt/IT: 84 → 86.4; BCom/IT: 84 → 84.4)
- Wrong application dates ("30 November" → "23 Feb 2026" / "27 Jul 2026")
- Wrong policy (BIT/Arts was stated to include an IT major - corrected to no IT major)
- Wrong dual-degree major policy (from 2026: only 1 major per program)
- Wrong fees (SSAF $365 → $373; living costs updated)
- Verbose answers condensed to 1–2 sentences

This file is the **validated ground truth** used as few-shot examples by the v2 generation script.

### `generated-qa-combined.csv` - Final fine-tuning dataset (2043 pairs) ✅
The primary dataset for fine-tuning. Produced by `generate_qa_extra.py` in three steps:
1. **Clean** - em dashes removed from the 1799 v2 pairs (replaced with hyphens)
2. **Top-up** - 300 new pairs generated (25 per topic), deduplicated against existing; 244 unique pairs retained
3. **Combine** - 1799 cleaned + 244 new = 2043 unique pairs after full deduplication

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

### `generated-qa-2000.csv` - v2 raw output (1799 unique pairs)
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

Prepared splits live in `data/` - see [`data/README.md`](data/README.md) for format and usage.

| Model | Method | Directory |
|-------|--------|-----------|
| Gemma 3 12B | GRPO + LoRA (BF16) + G-Eval reward | `fine-tuning/gemma3-12b-grpo/` |
| Mistral Nemo 12B | GRPO + LoRA (BF16) + G-Eval reward | `fine-tuning/mistral-nemo-12b-grpo/` |
| Qwen3 14B | GRPO + LoRA (BF16) + G-Eval reward | `fine-tuning/qwen3-14B-grpo/` |

Each model is evaluated across **16 configurations**: base + fine-tuned × 8 prompt variants (plain / system prompt / few-shot / both - each with and without RAG). G-Eval scores completions via **OpenAI GPT-4o-mini** on four dimensions (factual accuracy 55%, relevance 25%, conciseness 10%, no-hallucination 10%). W&B project: `uq-unibot / uni-bot`.

See the README inside each fine-tuning directory for setup and running instructions.

## RAG Evaluation Variant

Each of the 4 prompt variants has a RAG-enabled twin that retrieves relevant passages from UQ source documents before generating a response. The RAG pipeline follows SOTA practices (early 2026):

| Step | Technique | Reference |
|------|-----------|-----------|
| Ingestion | Contextual Retrieval - GPT-4o-mini generates a 1-sentence context prefix per chunk before embedding | Anthropic (2024), "Introducing Contextual Retrieval" - [anthropic.com/news/contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval) - reports 49% fewer retrieval failures |
| Embeddings | BGE-M3 multi-vector dense embedder (MTEB-leading open model) | Chen et al. (2024), "BGE M3-Embedding" - [arXiv:2402.03216](https://arxiv.org/abs/2402.03216) |
| Sparse retrieval | BM25 (rank_bm25) over the same contextualised chunks | - |
| Fusion | Reciprocal Rank Fusion of dense + sparse lists | Cormack, Clarke & Büttcher (2009), "Reciprocal Rank Fusion outperforms Condorcet…" - SIGIR 2009 |
| Reranking | `bge-reranker-v2-m3` cross-encoder on the fused top-20 | Li et al. (2023), "Making Large Language Models A Better Foundation For Dense Retrieval" - [arXiv:2312.15503](https://arxiv.org/abs/2312.15503) |

The index is built once and shared across all models and variants in a single evaluation run.

### Building the RAG index

```powershell
# Install RAG dependencies
pip install -r requirements.txt

# Build index from data-collection/sources/*.pdf + program-page URLs
python -m rag build
```

Outputs to `rag/index/` (gitignored):
- `chunks.jsonl` - chunk text + context prefix + source metadata
- `dense.faiss` - FAISS IndexFlatIP for dense retrieval
- `bm25.pkl` - serialised BM25Okapi index

Adding a new PDF (e.g., the Domestic Student Guide) is automatic - drop it into `data-collection/sources/` and re-run `python -m rag build`.

### Querying the index

```powershell
python -m rag query "What is the minimum ATAR for the BIT?"
```

### Running evaluation with/without RAG

```powershell
# Full 16-config run (default - requires RAG index to be built first)
python fine-tuning/gemma3-12b-grpo/evaluate.py

# Skip RAG variants - 8-config non-RAG run only
python fine-tuning/gemma3-12b-grpo/evaluate.py --no-rag

# Use a custom index location
python fine-tuning/gemma3-12b-grpo/evaluate.py --index-dir /path/to/rag/index
```

If the index is not built, or RAG dependencies are not installed, the RAG variants are skipped automatically with a warning - the non-RAG variants still run.

---

## Source Documents

| File | Description |
|------|-------------|
| `sources/program-pages.txt` | Official UQ program page URLs for BIT and 7 dual degrees |
| `sources/international-guide-undergraduate-postgraduate.pdf` | UQ International Student Guide 2026 - primary factual reference |
| `sources/domestic-guide-undergraduate.pdf` | UQ Domestic Undergraduate Student Guide 2026 |
