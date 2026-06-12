# Data Collection

> **Anonymisation notice:** All files in this directory have been de-identified for double-blind review. The institution name, URLs, program codes, CRICOS codes, course-code prefixes, named programs, campus names, and source PDFs have been replaced with generic placeholders. See the root `README.md` for the full substitution table. The source PDFs (`international-guide-undergraduate-postgraduate.pdf`, `domestic-guide-undergraduate.pdf`) have been removed; drop your institution's equivalents into `sources/` to rebuild the RAG index.

This folder contains the question-and-answer datasets used in the LLM student query comparison research project. The datasets cover student queries about the University (the University) Bachelor of Information Technology (BIT) degree and its related dual degrees.

## Datasets

| File | Pairs | Description |
|------|-------|-------------|
| `manual/data-manual.xlsx` | 38 | Human-authored baseline Q&A pairs |
| `generated/generated-qa-200.xlsx` | 200 | AI-generated synthetic Q&A pairs |

Both files share the same two-column format: **Question** (column A) and **Answer** (column B).

## How the 200 Generated Questions Were Created

The 200 synthetic Q&A pairs in `generated/generated-qa-200.xlsx` were produced using **Claude Code** (Anthropic, `claude-sonnet-4-6`) in an interactive session in May 2026. The process involved the following steps:

### 1. Source Gathering

Claude Code identified and compiled all relevant the University official program pages for BIT and dual degrees into `sources/program-pages.txt`. This covered 8 programs including the standalone BIT (9004), BIT (Honours) (9001), and dual degrees with Business Management, Commerce, Human Movement and Nutrition Sciences, Arts, Engineering, and Design.

### 2. Source Reading

Claude Code read three primary sources to ground all answers in verified information:

- **`sources/international-guide-undergraduate-postgraduate.pdf`** - the University International Student Guide 2026, covering entry requirements, fees, English proficiency, visas, and campus life.
- **`sources/program-pages.txt`** - the University program page URLs for BIT and related dual degrees.
- **`manual/data-manual.xlsx`** - the existing human-authored dataset, used to understand the required answer format and avoid duplicating questions.

### 3. Question Design

Questions were designed to reflect the kinds of queries real domestic and international students ask, informed by Reddit/student forum discussions. They were distributed across 12 topic categories to ensure broad coverage:

- Program overview and structure
- Domestic entry requirements and ATAR
- International entry requirements
- English language proficiency
- Tuition fees and financial support
- Student visas and work rights
- Course enrollment and study planning
- Majors and elective choices
- Core courses (CORE1001, CORE2002, INFO1200, INFO2200, DSGN1400, DSGN3801, MATH1061)
- Grading, GPA, and academic standing
- Student support and wellbeing
- Credit, recognition of prior learning, and post-graduation pathways

### 4. Script Generation and Execution

Claude Code wrote `generated/generate_qa.py`, a Python script using `openpyxl` that encodes all 200 Q&A pairs and exports them to a formatted Excel workbook matching the style of the manual dataset. The script was then executed to produce `generated/generated-qa-200.xlsx`.

## Reproducing the Generated Dataset

```bash
pip install openpyxl
python data-collection/generated/generate_qa.py
```

## Sources

- the University program pages: [study.example.edu](https://study.example.edu) and [programs-courses.example.edu](https://programs-courses.example.edu)
- the University International Student Guide 2026 (PDF)
- Reddit/student forums (for question style reference)
