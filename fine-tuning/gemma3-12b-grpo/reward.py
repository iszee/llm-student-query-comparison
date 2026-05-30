"""
reward.py
---------
G-Eval reward function for GRPO fine-tuning.

Scores each model completion against the reference answer using OpenAI GPT
on four rubric dimensions (factual accuracy, relevance, conciseness,
no-hallucination), returning a scalar reward in [0, 1].

Requires: OPENAI_API_KEY environment variable.

Standalone smoke test:
    python fine-tuning/gemma3-12b-grpo/reward.py
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from config import Config

# ── G-Eval prompt template ────────────────────────────────────────────────────

GEVAL_TEMPLATE = """\
You are an impartial evaluator assessing the quality of an AI assistant's answer \
to a student question about the UQ Bachelor of Information Technology program.

Score the model answer on FOUR criteria, each on an integer scale from 0 to 5.

Question:         {question}
Reference answer: {reference}
Model answer:     {prediction}

Criteria:
1. Factual accuracy    — Is the model answer factually correct about the UQ BIT program?
   The reference shows ONE acceptable answer — do NOT penalise correct information that is
   phrased differently from or goes beyond the reference. Only penalise claims that are
   factually wrong or directly contradict the reference/question.
   (0.0 = clearly wrong or contradicts facts, 5.0 = fully correct)
2. Relevance           — Does the answer directly address the question asked?
   (0.0 = off-topic or ignores the question entirely, 5.0 = precisely on-point)
3. Conciseness         — Is the length appropriate for the question? A detailed answer to a
   broad question should not be penalised. Penalise only unnecessary repetition or padding.
   (0.0 = grossly padded/repetitive or severely truncated, 5.0 = just right)
4. No hallucination    — Does the answer avoid stating uncertain facts as certain?
   (0.0 = fabricates details not supported by the reference, 5.0 = never invents information)

Reply ONLY with a valid JSON object containing a brief reasoning note and the four scores:
{{"reasoning": "<one sentence>", "factual_accuracy": <int 0-5>, "relevance": <int 0-5>, "conciseness": <int 0-5>, "no_hallucination": <int 0-5>}}
"""

# ── Scorer ────────────────────────────────────────────────────────────────────

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from env
    return _client


def geval_score(question: str, reference: str, prediction: str, cfg: Config) -> float:
    """
    Call OpenAI GPT to score a single completion, optionally averaging N samples.

    When cfg.geval_samples == 1 (default): single deterministic call (temperature=0).
    When cfg.geval_samples > 1: N calls at cfg.geval_sample_temperature; per-dimension
    scores are averaged before computing the composite (follows Liu et al. 2023 G-Eval).

    Returns a scalar reward in [0, 1]. Returns 0.0 if all samples fail.
    """
    prompt = GEVAL_TEMPLATE.format(
        question=question,
        reference=reference,
        prediction=prediction,
    )
    client = _get_client()
    n_samples = getattr(cfg, "geval_samples", 1)
    temperature = 0.0 if n_samples <= 1 else getattr(cfg, "geval_sample_temperature", 0.5)

    dim_accum = {dim: 0.0 for dim in cfg.geval_weights}
    successful = 0

    for s in range(n_samples):
        for attempt in range(cfg.geval_max_retries):
            try:
                resp = client.chat.completions.create(
                    model=cfg.geval_model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=cfg.geval_timeout,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=400,
                )
                scores = json.loads(resp.choices[0].message.content)
                for dim in cfg.geval_weights:
                    val = max(0.0, min(5.0, float(scores.get(dim, 0))))
                    dim_accum[dim] += val
                successful += 1
                break   # sample succeeded — move to next sample
            except Exception as exc:
                wait = 2 ** attempt
                print(f"[reward] G-Eval sample {s+1}/{n_samples} attempt {attempt+1} failed: {exc}. "
                      f"Retrying in {wait}s...")
                time.sleep(wait)

    if successful == 0:
        print("[reward] All G-Eval samples exhausted — returning 0.0")
        return 0.0

    # Average dims across successful samples, then compute composite
    weighted = sum((dim_accum[dim] / successful) * w for dim, w in cfg.geval_weights.items())
    return weighted / 5.0


# ── Batch reward function (called by GRPOTrainer) ────────────────────────────

def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    Reward function signature expected by TRL GRPOTrainer.

    GRPOTrainer calls this with:
      - prompts:     formatted input strings (one per completion)
      - completions: decoded model output strings
      - **kwargs:    dataset columns passed through (we use "reference" and "question")

    Returns a list of scalar rewards, one per completion.

    Calls are parallelised with ThreadPoolExecutor (geval_max_workers threads) because
    each geval_score call is I/O-bound (HTTPS to OpenAI). With 64 calls per step and
    16 workers this reduces scoring from ~96 s to ~6 s per optimizer step.
    """
    cfg: Config = kwargs.get("cfg", Config())
    questions: list[str] = kwargs.get("question", [""] * len(completions))
    references: list[str] = kwargs.get("reference", [""] * len(completions))

    with ThreadPoolExecutor(max_workers=cfg.geval_max_workers) as pool:
        futures = {
            pool.submit(geval_score, q, ref, pred, cfg): i
            for i, (q, ref, pred) in enumerate(zip(questions, references, completions))
        }
        rewards: list[float] = [0.0] * len(completions)
        for future in as_completed(futures):
            idx = futures[future]
            rewards[idx] = future.result()

    return rewards


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Export it before running.")
        raise SystemExit(1)

    cfg = Config()

    test_cases = [
        {
            "question":   "What is the minimum ATAR for the BIT at UQ?",
            "reference":  "The minimum entry score for the BIT (program 2570) is 81.9 "
                          "for Semester 1, 2026 entry.",
            "prediction": "The minimum ATAR for the Bachelor of Information Technology "
                          "at UQ is 81.9 for Semester 1, 2026.",
            "expected":   "high (accurate, relevant, concise)",
        },
        {
            "question":   "What is the minimum ATAR for the BIT at UQ?",
            "reference":  "The minimum entry score for the BIT (program 2570) is 81.9 "
                          "for Semester 1, 2026 entry.",
            "prediction": "I'm not sure about the exact ATAR. You should probably check "
                          "the UQ website or contact admissions.",
            "expected":   "mid (safe but uninformative)",
        },
        {
            "question":   "What is the minimum ATAR for the BIT at UQ?",
            "reference":  "The minimum entry score for the BIT (program 2570) is 81.9 "
                          "for Semester 1, 2026 entry.",
            "prediction": "The minimum ATAR is 95. You will also need to pass an "
                          "interview and submit a portfolio.",
            "expected":   "low (factually wrong, hallucinated requirements)",
        },
    ]

    print(f"Smoke-testing G-Eval reward with model: {cfg.geval_model}\n")
    for i, tc in enumerate(test_cases, 1):
        score = geval_score(tc["question"], tc["reference"], tc["prediction"], cfg)
        print(f"Case {i} ({tc['expected']}): reward = {score:.4f}")

    print("\nSmoke test complete.")
