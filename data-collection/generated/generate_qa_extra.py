"""
generate_qa_extra.py
Three steps in one pass:
  1. Clean em dashes from generated-qa-2000.csv (replace — with -)
  2. Generate 300 new Q&A pairs (25 per topic, 12 topics)
  3. Combine + deduplicate → generated-qa-combined.csv

Usage:
  set ANTHROPIC_API_KEY=sk-ant-...
  python generate_qa_extra.py
"""

import anthropic
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
EXISTING_CSV = SCRIPT_DIR / "generated-qa-2000.csv"
CORRECTED_CSV = SCRIPT_DIR / "corrected" / "corrected-qa.csv"
EXTRA_CSV = SCRIPT_DIR / "generated-qa-extra-300.csv"
COMBINED_CSV = SCRIPT_DIR / "generated-qa-combined.csv"
PROGRESS_FILE = SCRIPT_DIR / "generated-qa-extra-progress.json"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 25  # one batch per topic

TOPIC_PLAN = [
    ("Domestic Admission and Entry Requirements", 25),
    ("International Admission and Entry Requirements", 25),
    ("Program Structure and Duration", 25),
    ("Majors and Specialisations", 25),
    ("Specific Courses and Study Plans", 25),
    ("Dual Degree Programs", 25),
    ("Fees and Financial Matters", 25),
    ("Campus Life and Student Support", 25),
    ("Career Outcomes and Pathways", 25),
    ("Honours Program", 25),
    ("Application Process", 25),
    ("Additional Topics (AI in BIT, accreditation, visa, exchange, integrity, comparison with BCS)", 25),
]
assert sum(n for _, n in TOPIC_PLAN) == 300

RULES = """
CRITICAL RULES - FOLLOW WITHOUT EXCEPTION:

1. Only state facts explicitly found in the provided UQ International Guide 2026
   or the validated Q&A examples. Never invent or extrapolate.

2. Never guess or make up: ATAR scores, fee amounts, dates, URLs, course codes,
   scholarship names, or visa requirements.

3. Never write "adjusted score" - write "minimum entry score" or state the number.

4. Apply every correction below exactly as written:
   - BIT (2570) minimum entry score for Semester 1, 2026: 81.9 (median 87.4, highest 95.45)
   - BBusMgt/IT minimum entry score: 86.4 (NOT 84)
   - BCom/IT minimum entry score: 84.4 (NOT 84)
   - BE(Hons)/IT minimum entry score: 84
   - Semester 1, 2026 commencement: 23 Feb 2026
   - Semester 2, 2026 commencement: 27 Jul 2026
   - BIT/Arts dual degree does NOT include an IT major (from 2026)
   - From 2026, dual degree students may only complete 1 major toward the single program;
     majors are no longer possible in dual degrees
   - SSAF maximum annual fee: AUD $373 for 2026
   - Living costs off-campus: AUD $2,026-$4,151/month
   - On-campus residential college: AUD $2,635-$4,168/month (meals included)
   - ACS accreditation: Yes, the BIT IS accredited by the Australian Computer Society
   - BIT (Honours) minimum GPA: 5.0 on UQ's 7-point scale
   - BIT (Honours) application deadlines:
       S1 international 30 Nov, S1 domestic 31 Jan
       S2 international 31 May, S2 domestic 30 Jun
   - BIT total units to graduate: 48 units over 3 years
   - International tuition 2026: AUD $58,056/year
   - Student visa work rights: 48 hours per fortnight during semester (from 1 Jul 2023)
   - Student visa minimum living cost evidence: AUD $29,710

5. Keep answers concise: 1-3 sentences. Use a short list only when it genuinely
   helps (e.g. listing QTAC codes or prerequisite courses).

6. If a specific fact is not confirmed in the source documents, direct the student
   to study.uq.edu.au rather than guessing.

7. Vary question phrasing: direct, conversational, hypothetical ("What if I..."),
   conditional ("If I don't have..."), first-person ("I am applying..."),
   comparing options ("Should I choose X or Y?").

8. Do not repeat questions already in the validated examples.

9. PUNCTUATION RULES - STRICTLY ENFORCED:
   - NEVER use em dashes (the long dash character).
   - Use a regular hyphen (-) where a dash is needed.
   - Use commas, colons, or full stops instead of em dashes.
   - Example correct: "The deadline is 30 November - apply early."
   - Example correct: "For Semester 1, the deadline is 30 November."
   - Example WRONG: "The deadline is 30 November - apply early." (with em dash)
"""


# ── Step 1: Clean em dashes ────────────────────────────────────────────────────

def clean_em_dashes(text: str) -> str:
    """Replace em dash (U+2014) with a hyphen, collapsing surrounding spaces."""
    # "word — word" → "word - word"
    # "word—word"   → "word - word"
    return re.sub(r"\s*—\s*", " - ", text).strip()


def load_and_clean_existing() -> list[tuple[str, str]]:
    """Load existing CSV, clean em dashes, return (question, answer) pairs."""
    pairs = []
    with open(EXISTING_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = clean_em_dashes(row.get("Question", "").strip())
            a = clean_em_dashes(row.get("Answer", "").strip())
            if q and a:
                pairs.append((q, a))
    return pairs


# ── Step 2: Generate 300 more ─────────────────────────────────────────────────

def load_corrected_csv() -> list[tuple[str, str]]:
    pairs = []
    with open(CORRECTED_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("Question", "").strip()
            a = row.get("Answer", "").strip()
            if q and a:
                pairs.append((q, a))
    return pairs


def format_examples(pairs: list[tuple[str, str]], max_examples: int = 30) -> str:
    step = max(1, len(pairs) // max_examples)
    sampled = pairs[::step][:max_examples]
    lines = [
        "VALIDATED Q&A EXAMPLES - use as fact and style reference.",
        "Do not repeat these questions. Do not contradict these answers.\n",
    ]
    for i, (q, a) in enumerate(sampled, 1):
        lines.append(f"Q{i}: {q}")
        lines.append(f"A{i}: {a}\n")
    return "\n".join(lines)


def generate_batch(
    client: anthropic.Anthropic,
    examples_text: str,
    topic_name: str,
    n_pairs: int,
) -> list[tuple[str, str]]:
    topic_prompt = (
        f"Generate exactly {n_pairs} new Q&A pairs about:\n"
        f"TOPIC: {topic_name}\n\n"
        "Cover different sub-aspects. Each question must sound like something a real "
        "prospective or current UQ BIT student would type.\n\n"
        "Output ONLY a valid JSON array - no markdown, no explanation:\n"
        '[{"q": "...", "a": "..."}, ...]'
    )
    dynamic_block = f"{RULES}\n\n{topic_prompt}"

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": examples_text,
                                "cache_control": {"type": "ephemeral"},
                            },
                            {
                                "type": "text",
                                "text": dynamic_block,
                            },
                        ],
                    }
                ],
            )
            raw = response.content[0].text.strip()
            if "```" in raw:
                start = raw.index("[")
                end = raw.rindex("]") + 1
                raw = raw[start:end]
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("Response JSON is not a list")
            result = []
            for item in data:
                q = clean_em_dashes(str(item.get("q", "")).strip())
                a = clean_em_dashes(str(item.get("a", "")).strip())
                if q and a:
                    result.append((q, a))
            return result
        except Exception as exc:
            wait = 15 * (attempt + 1)
            print(f"    Attempt {attempt + 1}/3 failed: {exc}. Retrying in {wait}s...")
            time.sleep(wait)

    print(f"    WARNING: All retries failed for '{topic_name}'. Skipping batch.")
    return []


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        return {"done": data.get("done", [])}
    return {"done": []}


def save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── Step 3: Combine + deduplicate ─────────────────────────────────────────────

def combine_and_dedup(
    existing: list[tuple[str, str]],
    extra: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    seen: set[str] = set()
    combined: list[tuple[str, str]] = []
    for q, a in existing + extra:
        key = q.strip().lower()
        if key not in seen:
            seen.add(key)
            combined.append((q, a))
    return combined


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set the ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    if not EXISTING_CSV.exists():
        print(f"ERROR: {EXISTING_CSV} not found.")
        sys.exit(1)

    if not CORRECTED_CSV.exists():
        print(f"ERROR: {CORRECTED_CSV} not found.")
        sys.exit(1)

    print("=== UQ BIT Q&A Extra Generator ===")
    print()

    # ── Step 1: Clean existing ────────────────────────────────────────────────
    print("Step 1: Loading and cleaning existing CSV (removing em dashes)...")
    existing_pairs = load_and_clean_existing()
    em_fixed = sum(1 for q, a in existing_pairs if "-" in q + a)
    print(f"  {len(existing_pairs)} pairs loaded, em dashes replaced with hyphens")
    print()

    # ── Step 2: Generate 300 ─────────────────────────────────────────────────
    print("Step 2: Generating 300 new Q&A pairs (25 per topic)...")
    corrected = load_corrected_csv()
    examples_text = format_examples(corrected)
    client = anthropic.Anthropic(api_key=api_key)
    progress = load_progress()
    done_keys: set[str] = set(progress["done"])

    # Build seen set from existing (to deduplicate during generation)
    seen_questions: set[str] = {q.strip().lower() for q, _ in existing_pairs}

    # Resume: load already-generated extras if resuming
    extra_pairs: list[tuple[str, str]] = []
    if EXTRA_CSV.exists() and done_keys:
        with open(EXTRA_CSV, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row.get("Question", "").strip()
                a = row.get("Answer", "").strip()
                if q and a:
                    extra_pairs.append((q, a))
                    seen_questions.add(q.lower())
        print(f"  Resuming: {len(extra_pairs)} extra pairs already generated")

    extra_file = open(
        EXTRA_CSV,
        "a" if (EXTRA_CSV.exists() and done_keys) else "w",
        newline="",
        encoding="utf-8",
    )
    extra_writer = csv.writer(extra_file)
    if not (EXTRA_CSV.exists() and done_keys):
        extra_writer.writerow(["#", "Question", "Answer"])
        extra_file.flush()

    total_batches = len(TOPIC_PLAN)
    total_extra_written = len(extra_pairs)
    total_extra_received = 0
    total_extra_skipped_dup = 0
    next_extra_row = total_extra_written + 1

    try:
        for b_idx, (topic_name, n_pairs) in enumerate(TOPIC_PLAN):
            key = f"{topic_name}::extra"
            if key in done_keys:
                print(f"[{b_idx + 1:2d}/{total_batches}] SKIP  {topic_name[:60]}")
                continue

            print(f"[{b_idx + 1:2d}/{total_batches}] GEN   {topic_name[:60]} ({n_pairs} pairs)...")

            new_pairs = generate_batch(client, examples_text, topic_name, n_pairs)
            total_extra_received += len(new_pairs)

            written_this = 0
            for q, a in new_pairs:
                key_q = q.strip().lower()
                if key_q not in seen_questions:
                    seen_questions.add(key_q)
                    extra_writer.writerow([next_extra_row, q, a])
                    extra_pairs.append((q, a))
                    next_extra_row += 1
                    written_this += 1
                else:
                    total_extra_skipped_dup += 1

            extra_file.flush()
            total_extra_written += written_this
            print(f"          -> {len(new_pairs)} received, {written_this} written (total extra: {total_extra_written})")

            if new_pairs:
                done_keys.add(key)
                progress["done"] = list(done_keys)
                save_progress(progress)

            if b_idx < total_batches - 1:
                time.sleep(8)

    finally:
        extra_file.close()

    print()
    print(f"  Extra generation done: {total_extra_received} received, "
          f"{total_extra_written} written, {total_extra_skipped_dup} dups skipped")
    print()

    # ── Step 3: Combine + deduplicate ─────────────────────────────────────────
    print("Step 3: Combining and deduplicating...")
    combined = combine_and_dedup(existing_pairs, extra_pairs)
    print(f"  Existing (cleaned): {len(existing_pairs)}")
    print(f"  Extra (new):        {len(extra_pairs)}")
    print(f"  Combined unique:    {len(combined)}")

    with open(COMBINED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["#", "Question", "Answer"])
        for i, (q, a) in enumerate(combined, 1):
            writer.writerow([i, q, a])

    print(f"  Saved to: {COMBINED_CSV}")
    print()
    print("Done.")

    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print("Progress file removed.")


if __name__ == "__main__":
    main()
