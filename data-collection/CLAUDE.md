# CLAUDE.md — data-collection

This file documents the AI-assisted data collection process for the llm-student-query-comparison research project.

## AI Tool Usage

### Round 1 — v1 synthetic dataset (May 2026)

The initial synthetic Q&A dataset (`generated/generated-qa-200.xlsx`) was created with the assistance of **Claude Code** (Anthropic, model: claude-sonnet-4-6) via an interactive session.

Claude Code performed the following tasks:

1. **Source identification** — searched UQ official program pages to compile all Bachelor of Information Technology and dual-degree program URLs, saved to `sources/program-pages.txt`.
2. **Source reading** — read the UQ International Guide 2026 PDF and the manual dataset to extract verified facts.
3. **Q&A authoring** — hardcoded 200 synthetic Q&A pairs into `generated/generate_qa.py`.
4. **Script generation** — wrote `generated/generate_qa.py`, producing formatted Excel output.
5. **Dataset generation** — executed the script to produce `generated/generated-qa-200.xlsx`.

### Round 2 — human correction (May 2026)

The 200 v1 pairs were manually reviewed. 17 were deleted; 183 were retained with corrections. Key errors found:

| Error type | Example |
|------------|---------|
| Wrong ATAR | BBusMgt/IT stated as 84, corrected to 86.4 |
| Wrong dates | S1 deadline "30 November" corrected to "23 Feb 2026" |
| Wrong policy | BIT/Arts said to include IT major — reversed to NO |
| Wrong policy | Dual degree majors — corrected to 1 major max from 2026 |
| Wrong fees | SSAF $365 → $373; living costs updated |
| Overly verbose | Many 4-sentence answers cut to 1–2 sentences |

The corrected dataset is saved as `generated/corrected/corrected-qa.csv` and is the **validated ground truth** for all subsequent generation.

### Round 3 — v2 fine-tuning dataset (May 2026)

2000 new Q&A pairs generated using `generated/generate_qa_v2.py`, which calls the Claude API (claude-sonnet-4-6) with:
- 30 representative corrected pairs as few-shot examples (prompt-cached for cost efficiency)
- Explicit anti-hallucination rules encoding every known correction from Round 2
- Incremental CSV writing and resume-on-interrupt via a progress JSON file

After deduplication: 1799 unique pairs written to `generated/generated-qa-2000.csv`.

### Round 4 — clean, top-up, and combine (May 2026)

`generated/generate_qa_extra.py` performed three steps in one pass:

1. **Clean** — loaded the 1799 v2 pairs and replaced all em dashes (—) with hyphens to normalise punctuation
2. **Top-up** — generated 300 new pairs (25 per topic × 12 topics), deduplicated against existing; 244 unique pairs written to `generated/generated-qa-extra-300.csv`
3. **Combine** — merged cleaned v2 pairs + new extras, final dedup → 2043 unique pairs in `generated/generated-qa-combined.csv`

The RULES block in `generate_qa_extra.py` explicitly bans em dashes so new pairs are clean by construction.

## Dataset Description

| File | Pairs | Format | Description |
|------|-------|--------|-------------|
| `manual/data-manual.xlsx` | 38 | Excel | Human-authored baseline Q&A pairs |
| `generated/generated-qa-200.xlsx` | 200 | Excel | v1 AI-generated pairs (contains errors — superseded) |
| `generated/corrected/corrected-qa.csv` | 183 | CSV | Human-corrected ground truth (validated) |
| `generated/generated-qa-2000.csv` | 1799 | CSV | v2 raw output (intermediate — superseded by combined) |
| `generated/generated-qa-extra-300.csv` | 244 | CSV | Top-up pairs from Round 4 (intermediate) |
| `generated/generated-qa-combined.csv` | **2043** | CSV | **Final fine-tuning dataset** (cleaned, deduplicated) |

## Topic Coverage (final combined dataset — 2043 pairs)

| Topic | v2 | Extra | Total |
|-------|-----|-------|-------|
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
| Additional Topics (AI, accreditation, visa, integrity, BIT vs BCS) | 200 | 20 | 220 |
| **Total** | **1799** | **244** | **2043** |

## Sources Used

- UQ International Student Guide 2026 (`sources/international-guide-undergraduate-postgraduate.pdf`)
- UQ program pages: study.uq.edu.au and programs-courses.uq.edu.au (see `sources/program-pages.txt`)
- Manual dataset (`manual/data-manual.xlsx`) — human baseline
- Corrected dataset (`generated/corrected/corrected-qa.csv`) — validated ground truth for generation

## Reproducibility

### Final combined dataset (recommended)

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
.\env\Scripts\Activate.ps1
python data-collection/generated/generate_qa_extra.py
```

Cleans `generated-qa-2000.csv`, generates 300 top-up pairs, combines and deduplicates to produce `generated-qa-combined.csv`. Resumes on interrupt. Estimated cost: ~$0.50-1 USD.

### v2 base dataset only

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python data-collection/generated/generate_qa_v2.py
```

Generates 2000 pairs across 104 batches (8s sleep between calls). Resumes on interrupt. Estimated cost: ~$3-6 USD.

### v1 dataset (no API needed, contains known errors)

```powershell
python data-collection/generated/generate_qa.py
```

Requires: `openpyxl` (`pip install openpyxl`)
