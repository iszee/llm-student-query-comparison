# Expert Validation Study — Runbook

## Files in this directory

| File | Purpose |
|------|---------|
| `build_study.py` | Master build script. Run once to regenerate `answer_key.csv` and `questionnaire.xlsx`. |
| `answer_key.csv` | **PRIVATE.** Maps every opaque item ID to its model/condition. Never share with raters. |
| `questionnaire.xlsx` | Rater-facing workbook. Distribute one copy per expert; they fill it and return it. |
| `join_responses.py` | Joins returned workbooks with `answer_key.csv` to produce a tidy analysis table. |
| `build_form.gs` | *(deprecated — Google Form was replaced by Excel due to unresponsive browser rendering)* |

All randomisation uses **seed = 42** (documented in `build_study.py`).

---

## Phase 1 — Build (one-time setup)

### 1a. Regenerate artefacts (if needed)

```powershell
# From repo root, with the virtualenv active:
python study/build_study.py
```

This writes `study/answer_key.csv` (50 rows, PRIVATE) and `study/questionnaire.xlsx`.

### 1b. Verify the workbook before distributing

Open `study/questionnaire.xlsx` and check:

- **Start here** sheet: purpose, consent, instructions, name + email input cells (yellow).
- **Q01–Q10** sheets: each sheet shows the student question and reference answer compactly at top,
  then a grid of 5 answer rows (A–E) × 4 metric columns (Factual accuracy, Relevance,
  Conciseness, No hallucination). Every metric cell is yellow and editable.
- **Score cells accept 0–5 whole numbers only** — try entering 6 and verify the warning appears.
- **No model names or condition identifiers are visible anywhere in the workbook.**
- **`_MAP` sheet is hidden** (very-hidden; not accessible via right-click → Unhide in Excel).

---

## Phase 2 — Distribute to raters

1. Make **one copy** of `questionnaire.xlsx` per expert (e.g. `rater_jdoe.xlsx`, `rater_smith.xlsx`).
   Renaming is optional but makes returned files easier to track.
2. Email each expert their own copy.

**Suggested email text:**

> Please use the attached Excel file to complete the University BIT expert questionnaire (~40–60 minutes).
>
> **Instructions:**
> - Fill in your name and email on the **Start here** tab.
> - Work through tabs **Q01–Q10** (one student question per tab).
> - For each of the 5 AI-generated answers (A–E), enter a score from **0–5** in each yellow cell.
>   0 = lowest quality, 5 = highest quality.
> - A reference answer from official the University sources is shown at the top of each tab as a quality benchmark.
> - Save the file and **reply with the filled workbook attached**.
>
> All responses are anonymous. Model names and system identifiers are not shown.

---

## Phase 3 — Collect responses

1. Create a folder `study/returned/`.
2. Save each returned workbook into that folder (keep the filenames distinct).

---

## Phase 4 — Join and analyse

### 4a. Run the join script

```powershell
python study/join_responses.py `
    --responses-dir study/returned `
    --key           study/answer_key.csv `
    --out           study/responses_tidy.csv
```

The script prints:
- Number of workbooks found and items read per rater
- Total rows and missing score count
- Any item IDs not matched in the answer key

### 4b. Output format

`responses_tidy.csv` has **one row per (rater × item_id × dimension)**. Key columns:

| Column | Description |
|--------|-------------|
| `rater` | Rater email or name (from the Start here sheet) |
| `item_id` | Opaque answer ID, e.g. `Q03_A` |
| `dimension` | `FACT` / `REL` / `CONC` / `NOHAL` |
| `score` | Human rating 0–5 (blank if cell was left empty) |
| `model` | Model name (joined from answer key) |
| `state` | `base` or `finetuned` |
| `variant` | Prompt variant (e.g. `sysprompt_rag_fewshot`) |
| `condition` | `{state}_{variant}` |
| `composite_score_geval` | Original G-Eval composite score (0–1) for correlation analysis |
| `*_geval` | Per-dimension G-Eval scores |

### 4c. Suggested analysis steps

```python
import pandas as pd
from scipy.stats import spearmanr

df = pd.read_csv("study/responses_tidy.csv")

# 1. Rating items only
ratings = df.copy()

# 2. Mean human score per item per dimension (inter-rater average)
agg = ratings.groupby(["item_id", "dimension"])["score"].mean().reset_index()

# 3. Join G-Eval scores for correlation
key = pd.read_csv("study/answer_key.csv")
dim_map = {
    "FACT":  "factual_accuracy",
    "REL":   "relevance",
    "CONC":  "conciseness",
    "NOHAL": "no_hallucination",
}
rows = []
for dim_code, geval_col in dim_map.items():
    sub = agg[agg["dimension"] == dim_code].merge(
        key[["item_id", geval_col]], on="item_id"
    )
    r, p = spearmanr(sub["score"], sub[geval_col])
    rows.append({"dimension": dim_code, "spearman_r": r, "p_value": p})
print(pd.DataFrame(rows))
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `questionnaire.xlsx` missing `_MAP` sheet | Regenerate with `python study/build_study.py` — old copies lack the map |
| `join_responses.py` reports 0 items for a file | The file is missing the `_MAP` sheet (see above) |
| Rater name/email shows as file stem | Rater forgot to fill the Start here tab — ask them to resend |
| High missing-score count | Rater left cells blank; follow up with them |
| Score validation warning appeared but rater accepted it | Scores outside 0–5 are stored as-is; filter `df[df["score"].between(0,5)]` before analysis |
| Excel says workbook is protected | Raters should only type in yellow cells; other cells are locked by design |

---

## Phase 5 — Validation analysis

Run this phase after all rater workbooks have been returned and collected into `study/returned/`.

### 5a. Install extra dependencies

```powershell
pip install pingouin krippendorff matplotlib seaborn
```

### 5b. Join responses

```powershell
python study/join_responses.py `
    --responses-dir study/returned `
    --key           study/answer_key.csv `
    --out           study/responses_tidy.csv
```

### 5c. Run the analysis

```powershell
python study/analyze_validation.py
```

Outputs:

| File | Contents |
|------|----------|
| `study/validation_report.md` | Full narrative report: descriptive stats, inter-rater reliability, human vs G-Eval agreement, per-question ranking, implications for the paper |
| `study/agreement_stats.csv` | Machine-readable flat table of all statistics |
| `study/figures/scatter_human_vs_geval.png` | Per-dimension human consensus vs G-Eval scatter |
| `study/figures/composite_scatter.png` | Composite score scatter |
| `study/figures/bias_by_dimension.png` | G-Eval − Human bias per dimension |
| `study/figures/interrater_heatmap.png` | Pairwise Spearman heatmap (raters + G-Eval) |

### 5d. Interpret results

The report directly addresses **Limitation #1** in `PAPER_CONTEXT.md` (LLM-as-judge bias, no human evaluation). Key metrics:

- **ICC(2,k)** — inter-rater reliability of the expert panel (Koo & Mae 2016: ≥0.75 = good, ≥0.90 = excellent)
- **Krippendorff's α** — ordinal inter-rater agreement (≥0.80 = reliable, ≥0.67 = tentative)
- **Spearman ρ / Pearson r (human consensus vs G-Eval)** — the headline human–machine agreement
- **Systematic bias** — mean(G-Eval − Human) with Wilcoxon and t-test; positive = judge inflates
- **Per-question Spearman ρ** — whether G-Eval ranks the 5 answers the same way experts do

See Section 8 of `validation_report.md` for a ready-to-use summary of implications.

---

## Re-running build_study.py

The script is fully deterministic (seed 42). Re-running it will produce identical
`answer_key.csv` and `questionnaire.xlsx`, provided the source data files have not changed.
If you need to change the sample or letter assignments, change `SEED` in `build_study.py`
**and** document the new seed here.
