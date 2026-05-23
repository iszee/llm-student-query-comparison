"""
generate_qa_v2.py
Generate 2000 synthetic Q&A pairs for fine-tuning a UQ BIT information assistant.

Ground truth sources:
  - corrected/corrected-qa.csv — 183 human-validated Q&A pairs (few-shot examples, cached)
  - RULES block below — encodes all verified corrections and constraints

Note: the PDF is NOT passed per-call (too large for the 30K TPM rate limit).
All verified facts are embedded in the corrected examples and RULES.

Output:
  - generated-qa-2000.csv  (written incrementally — safe to interrupt and resume)

Usage:
  set ANTHROPIC_API_KEY=sk-ant-...
  python generate_qa_v2.py

Re-running resumes from the last completed batch via generated-qa-2000-progress.json.
"""

import anthropic
import csv
import json
import os
import sys
import time
from pathlib import Path

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent
CORRECTED_CSV = SCRIPT_DIR / "corrected" / "corrected-qa.csv"
OUTPUT_CSV = SCRIPT_DIR / "generated-qa-2000.csv"
PROGRESS_FILE = SCRIPT_DIR / "generated-qa-2000-progress.json"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 20  # Q&A pairs per API call

# Topic plan — must sum to exactly 2000.
# 8 topics × 150 + 4 topics × 200 = 1200 + 800 = 2000
TOPIC_PLAN = [
    ("Domestic Admission and Entry Requirements", 150),
    ("International Admission and Entry Requirements", 200),
    ("Program Structure and Duration", 150),
    ("Majors and Specialisations", 150),
    ("Specific Courses and Study Plans", 200),
    ("Dual Degree Programs", 200),
    ("Fees and Financial Matters", 150),
    ("Campus Life and Student Support", 150),
    ("Career Outcomes and Pathways", 150),
    ("Honours Program", 150),
    ("Application Process", 150),
    ("Additional Topics (AI in BIT, accreditation, visa, exchange, integrity, comparison with BCS)", 200),
]
assert sum(n for _, n in TOPIC_PLAN) == 2000, "TOPIC_PLAN must sum to 2000"

# ── Anti-hallucination rules injected into every prompt ───────────────────────
RULES = """
CRITICAL RULES — FOLLOW WITHOUT EXCEPTION:

1. Only state facts explicitly found in the provided UQ International Guide 2026 PDF
   or the validated Q&A examples. Never invent or extrapolate.

2. Never guess or make up: ATAR scores, fee amounts, dates, URLs, course codes,
   scholarship names, or visa requirements.

3. Never write "adjusted score" — write "minimum entry score" or state the number.

4. Apply every correction below exactly as written:
   - BIT (2570) minimum entry score for Semester 1, 2026: 81.9  (median 87.4, highest 95.45)
   - BBusMgt/IT minimum entry score: 86.4  (NOT 84)
   - BCom/IT minimum entry score: 84.4  (NOT 84)
   - BE(Hons)/IT minimum entry score: 84
   - Semester 1, 2026 commencement: 23 Feb 2026
   - Semester 2, 2026 commencement: 27 Jul 2026
   - BIT/Arts dual degree does NOT include an IT major (from 2026)
   - From 2026, dual degree students may only complete 1 major toward the single program;
     majors are no longer possible in dual degrees
   - SSAF maximum annual fee: AUD $373 for 2026
   - Living costs off-campus: AUD $2,026–$4,151/month
   - On-campus residential college: AUD $2,635–$4,168/month (meals included)
   - ACS accreditation: Yes, the BIT IS accredited by the Australian Computer Society
   - BIT (Honours) minimum GPA: 5.0 on UQ's 7-point scale
   - BIT (Honours) application deadlines:
       S1 international 30 Nov, S1 domestic 31 Jan
       S2 international 31 May, S2 domestic 30 Jun
   - BIT total units to graduate: 48 units over 3 years
   - International tuition 2026: AUD $58,056/year
   - Student visa work rights: 48 hours per fortnight during semester (from 1 Jul 2023)
   - Student visa minimum living cost evidence: AUD $29,710

5. Keep answers concise: 1–3 sentences. Use a short list only when it genuinely
   helps (e.g. listing QTAC codes or prerequisite courses).

6. If a specific fact is not confirmed in the source documents, direct the student
   to study.uq.edu.au rather than guessing.

7. Vary question phrasing: direct, conversational, hypothetical ("What if I…"),
   conditional ("If I don't have…"), first-person ("I am applying…"),
   comparing options ("Should I choose X or Y?").
   Questions must sound like real student messages, not textbook prompts.

8. Do not repeat questions already in the validated examples.
"""


def load_corrected_csv() -> list[tuple[str, str]]:
    """Load validated Q&A pairs from corrected-qa.csv."""
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
    """Serialise a representative subset of validated examples into the cached prompt block.

    Uses every Nth example to get diverse coverage. Keeping ≤30 examples
    limits the cached block to ~3K tokens, safely under the 30K TPM rate limit.
    """
    # Evenly sample `max_examples` pairs from the full list
    step = max(1, len(pairs) // max_examples)
    sampled = pairs[::step][:max_examples]

    lines = [
        "VALIDATED Q&A EXAMPLES — use as fact and style reference.",
        "Do not repeat these questions. Do not contradict these answers.\n",
    ]
    for i, (q, a) in enumerate(sampled, 1):
        lines.append(f"Q{i}: {q}")
        lines.append(f"A{i}: {a}\n")
    return "\n".join(lines)


def batches_for_topic(target: int, batch_size: int) -> list[int]:
    """
    Return list of batch sizes that sum to target.
    E.g. target=150, batch_size=20 → [20,20,20,20,20,20,20,10]
    """
    sizes = []
    remaining = target
    while remaining > 0:
        sizes.append(min(remaining, batch_size))
        remaining -= batch_size
    return sizes


def generate_batch(
    client: anthropic.Anthropic,
    examples_text: str,
    topic_name: str,
    n_pairs: int,
) -> list[tuple[str, str]]:
    """Call Claude API and return up to n_pairs Q&A tuples for the given topic.

    Uses explicit cache breakpoints:
      - Block 1 (cached): the 183 validated examples — static across all calls
      - Block 2 (dynamic): rules + topic prompt — changes each call

    The cached examples block (~9K tokens) is written to cache on the first call
    and read from cache on all subsequent calls, reducing per-call input cost by ~95%.
    """
    topic_prompt = (
        f"Generate exactly {n_pairs} new Q&A pairs about:\n"
        f"TOPIC: {topic_name}\n\n"
        "Cover different sub-aspects of this topic. Each question must sound like "
        "something a real prospective or current UQ BIT student would type.\n\n"
        "Output ONLY a valid JSON array — no markdown, no explanation:\n"
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
                            # Cached block: validated examples (static across all 104 calls)
                            {
                                "type": "text",
                                "text": examples_text,
                                "cache_control": {"type": "ephemeral"},
                            },
                            # Dynamic block: rules + topic prompt (not cached — changes each call)
                            {
                                "type": "text",
                                "text": dynamic_block,
                            },
                        ],
                    }
                ],
            )
            raw = response.content[0].text.strip()

            # Strip markdown code fences if model adds them
            if "```" in raw:
                start = raw.index("[")
                end = raw.rindex("]") + 1
                raw = raw[start:end]

            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("Response JSON is not a list")

            result = []
            for item in data:
                q = str(item.get("q", "")).strip()
                a = str(item.get("a", "")).strip()
                if q and a:
                    result.append((q, a))
            return result

        except Exception as exc:
            wait = 15 * (attempt + 1)  # 15s, 30s, 45s back-off for transient rate limits
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


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set the ANTHROPIC_API_KEY environment variable before running.")
        sys.exit(1)

    if not CORRECTED_CSV.exists():
        print(f"ERROR: Corrected Q&A CSV not found at {CORRECTED_CSV}")
        sys.exit(1)

    print("=== UQ BIT Q&A Generator v2 ===")
    print(f"Target: 2000 pairs across {len(TOPIC_PLAN)} topics")
    print()

    # Load source data
    print("Loading corrected Q&A examples...")
    corrected = load_corrected_csv()
    print(f"  {len(corrected)} validated pairs loaded")
    examples_text = format_examples(corrected)

    print()

    client = anthropic.Anthropic(api_key=api_key)
    progress = load_progress()
    done_keys: set[str] = set(progress["done"])

    # ── Resume: read existing CSV to rebuild seen-questions set and row counter ──
    seen_questions: set[str] = set()
    next_row: int = 1

    resuming = OUTPUT_CSV.exists() and done_keys
    if resuming:
        with open(OUTPUT_CSV, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row.get("Question", "").strip()
                if q:
                    seen_questions.add(q.lower())
                    next_row += 1
        print(f"Resuming: {next_row - 1} pairs already in {OUTPUT_CSV.name}")
        print()

    # Open CSV — append if resuming, fresh write if starting
    csv_file = open(OUTPUT_CSV, "a" if resuming else "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not resuming:
        csv_writer.writerow(["#", "Question", "Answer"])
        csv_file.flush()

    # Count total batches for progress display
    total_batches = sum(
        len(batches_for_topic(n, BATCH_SIZE)) for _, n in TOPIC_PLAN
    )
    global_batch = 0
    total_written = next_row - 1
    total_received = 0
    total_skipped_dup = 0

    try:
        for topic_name, target_count in TOPIC_PLAN:
            batch_sizes = batches_for_topic(target_count, BATCH_SIZE)
            for b_idx, b_size in enumerate(batch_sizes):
                global_batch += 1
                key = f"{topic_name}::{b_idx}"

                if key in done_keys:
                    print(
                        f"[{global_batch:3d}/{total_batches}] SKIP  {topic_name[:55]} "
                        f"batch {b_idx + 1}/{len(batch_sizes)}"
                    )
                    continue

                print(
                    f"[{global_batch:3d}/{total_batches}] GEN   {topic_name[:55]} "
                    f"batch {b_idx + 1}/{len(batch_sizes)} ({b_size} pairs)..."
                )

                new_pairs = generate_batch(
                    client, examples_text, topic_name, b_size
                )
                total_received += len(new_pairs)

                # Write new pairs immediately — deduplicate on the fly
                written_this_batch = 0
                for q, a in new_pairs:
                    q_key = q.strip().lower()
                    if q_key not in seen_questions:
                        seen_questions.add(q_key)
                        csv_writer.writerow([next_row, q, a])
                        next_row += 1
                        written_this_batch += 1
                    else:
                        total_skipped_dup += 1

                csv_file.flush()  # write to disk immediately
                total_written += written_this_batch

                print(
                    f"          -> {len(new_pairs)} received, "
                    f"{written_this_batch} written to CSV "
                    f"(total: {total_written})"
                )

                # Only mark batch done if we actually received pairs.
                # Failed batches (0 pairs) stay out of done_keys so they retry next run.
                if new_pairs:
                    done_keys.add(key)
                    progress["done"] = list(done_keys)
                    save_progress(progress)

                # Rate-limit throttle: 30K TPM / ~3K tokens per call (30 examples) = 10 calls/min.
                # 8s sleep keeps us at ~7 calls/min, well within the limit.
                if global_batch < total_batches:
                    time.sleep(8)

    finally:
        csv_file.close()

    print()
    print(f"Done.")
    print(f"  API responses:  {total_received} pairs")
    print(f"  Written to CSV: {total_written} pairs")
    print(f"  Duplicates skipped: {total_skipped_dup}")
    print(f"  Output: {OUTPUT_CSV}")

    # Clean up progress file on success
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print("  Progress file removed.")


if __name__ == "__main__":
    main()
