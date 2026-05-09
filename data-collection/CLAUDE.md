# CLAUDE.md — data-collection

This file documents the AI-assisted data collection process for the llm-student-query-comparison research project.

## AI Tool Usage

The synthetic Q&A dataset in `generated/generated-qa-200.xlsx` was created with the assistance of **Claude Code** (Anthropic, model: claude-sonnet-4-6) via an interactive session in May 2026.

Claude Code performed the following tasks in this directory:

1. **Source identification** — searched UQ official program pages to compile all Bachelor of Information Technology and dual-degree program URLs, saved to `sources/program-pages.txt`.
2. **Source reading** — read the UQ International Guide 2026 PDF (`sources/international-guide-undergraduate-postgraduate.pdf`) and the manual dataset (`manual/data-manual.xlsx`) to extract verified facts about programs, fees, entry requirements, and student policies.
3. **Q&A authoring** — authored 200 synthetic question-and-answer pairs covering 12 topic categories relevant to domestic and international BIT students at UQ.
4. **Script generation** — wrote `generated/generate_qa.py`, which produces the formatted Excel output using `openpyxl`.
5. **Dataset generation** — executed the script to produce `generated/generated-qa-200.xlsx`.

## Dataset Description

| File | Description |
|------|-------------|
| `manual/data-manual.xlsx` | 38 manually curated Q&A pairs (human-authored baseline) |
| `generated/generated-qa-200.xlsx` | 200 AI-assisted synthetic Q&A pairs |
| `generated/generate_qa.py` | Python script that produced the generated dataset |

### Topic Coverage (generated dataset)

The 200 Q&A pairs are distributed across:
- Program overview and structure (BIT and dual degrees)
- Entry requirements (domestic and international)
- Tuition fees and financial support
- English language requirements
- Visa and work rights for international students
- Course enrollment and academic planning
- Majors and elective choices
- Specific core courses (CSSE1001, CSSE2002, INFS1200, DECO1400, MATH1061, etc.)
- Grading, GPA, and academic standing
- Student support and wellbeing
- Credit and recognition of prior learning
- Post-graduation pathways

### Sources Used

- UQ International Student Guide 2026 (PDF)
- UQ program pages: study.uq.edu.au and programs-courses.uq.edu.au (see `sources/program-pages.txt`)
- Manual dataset (`manual/data-manual.xlsx`) — used for format reference and to avoid duplication
- Reddit/student forum queries — used to inform the style and topics of student questions

## Reproducibility

To regenerate the Excel file:

```bash
python data-collection/generated/generate_qa.py
```

Requires: `openpyxl` (`pip install openpyxl`)
