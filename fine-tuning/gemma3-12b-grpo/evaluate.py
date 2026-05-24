"""
evaluate.py
-----------
Evaluate a fine-tuned Gemma 3 12B LoRA checkpoint on the test set.

Generates a response for every example in test.jsonl, scores each with G-Eval
(OpenAI GPT-4o-mini), and saves results + aggregate metrics to a CSV.

Usage:
    # Evaluate the final checkpoint
    python fine-tuning/gemma3-12b-grpo/evaluate.py

    # Evaluate a specific mid-training checkpoint
    python fine-tuning/gemma3-12b-grpo/evaluate.py \\
        --checkpoint fine-tuning/gemma3-12b-grpo/checkpoints/checkpoint-50 \\
        --output fine-tuning/gemma3-12b-grpo/eval_ckpt50.csv

    # Dry run — generation only, skip G-Eval (no OpenAI cost)
    python fine-tuning/gemma3-12b-grpo/evaluate.py --no-geval

    # Evaluate the base model without any adapter
    python fine-tuning/gemma3-12b-grpo/evaluate.py --checkpoint none

Requires env vars:
    HF_TOKEN        — gated Gemma 3 weights on HuggingFace Hub
    OPENAI_API_KEY  — G-Eval scoring via OpenAI (not needed with --no-geval)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# Allow running from repo root or from within the fine-tuning directory
sys.path.insert(0, str(Path(__file__).parent))

# ── Lazy imports (heavy; only pulled in when the model actually loads) ────────
# Imported at the top of main() to avoid slow startup on --help.

from config import Config
from reward import GEVAL_TEMPLATE, _get_client  # reuse template + client singleton

# ── System prompt (must match train.py) ──────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful information assistant for the UQ Bachelor of Information "
    "Technology program. Answer student questions accurately and concisely. "
    "Only provide information you are confident about. "
    "If you are not certain about a specific fact, say so and direct the student "
    "to study.uq.edu.au rather than guessing."
)


# ── G-Eval (per-dimension) ────────────────────────────────────────────────────

def score_detailed(
    question: str,
    reference: str,
    prediction: str,
    cfg: Config,
) -> dict[str, float]:
    """
    Call OpenAI GPT-4o-mini to score a single completion.

    Returns a dict with keys:
        factual_accuracy, relevance, conciseness, no_hallucination  — raw 1-5 scores
        composite_score  — weighted average scaled [1,5] → [0,1]

    Returns all-1.0 raw scores (→ 0.0 composite) if all retries fail.
    """
    prompt = GEVAL_TEMPLATE.format(
        question=question,
        reference=reference,
        prediction=prediction,
    )
    client = _get_client()

    for attempt in range(cfg.geval_max_retries):
        try:
            resp = client.chat.completions.create(
                model=cfg.geval_model,
                messages=[{"role": "user", "content": prompt}],
                timeout=cfg.geval_timeout,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=64,
            )
            raw = json.loads(resp.choices[0].message.content)

            # Clamp and collect per-dimension scores
            dims = {}
            weighted = 0.0
            for dim, weight in cfg.geval_weights.items():
                val = float(raw.get(dim, 1))
                val = max(1.0, min(5.0, val))
                dims[dim] = val
                weighted += weight * val

            dims["composite_score"] = (weighted - 1.0) / 4.0
            return dims

        except Exception as exc:
            wait = 2 ** attempt
            print(f"[evaluate] G-Eval attempt {attempt + 1} failed: {exc}. "
                  f"Retrying in {wait}s...")
            time.sleep(wait)

    print("[evaluate] All G-Eval retries exhausted — recording zeros.")
    return {
        "factual_accuracy": 1.0,
        "relevance": 1.0,
        "conciseness": 1.0,
        "no_hallucination": 1.0,
        "composite_score": 0.0,
    }


def score_all_parallel(
    rows: list[dict],
    cfg: Config,
) -> list[dict[str, float]]:
    """
    Score a list of {question, reference, generated_response} dicts in parallel.
    Returns a list of score dicts aligned with `rows`.
    """
    results: list[dict[str, float]] = [{}] * len(rows)

    with ThreadPoolExecutor(max_workers=cfg.geval_max_workers) as pool:
        futures = {
            pool.submit(
                score_detailed,
                r["question"],
                r["reference"],
                r["generated_response"],
                cfg,
            ): i
            for i, r in enumerate(rows)
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="G-Eval scoring",
            unit="example",
        ):
            idx = futures[future]
            results[idx] = future.result()

    return results


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(checkpoint: str | None, cfg: Config):
    """
    Load base Gemma 3 12B in BF16 + optional LoRA adapter.

    Args:
        checkpoint: path to a PEFT adapter directory, or None/empty to use
                    the base model only.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import huggingface_hub

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        huggingface_hub.login(token=hf_token)

    print(f"Loading base model: {cfg.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if checkpoint and checkpoint.lower() != "none":
        adapter_path = Path(checkpoint)
        if not adapter_path.exists():
            print(f"WARNING: Checkpoint path not found: {adapter_path}. "
                  "Running base model only.")
        else:
            print(f"Loading LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model.eval()
    else:
        print("No adapter — evaluating base model.")

    return model, tokenizer


# ── Generation ────────────────────────────────────────────────────────────────

def generate_responses(
    examples: list[dict],
    model,
    tokenizer,
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    """
    Generate one response per example using the loaded model.

    Args:
        examples: list of dicts with at least a "question" key.
    Returns:
        List of decoded response strings (one per example).
    """
    responses: list[str] = []

    for i in tqdm(
        range(0, len(examples), batch_size),
        desc="Generating responses",
        unit="batch",
    ):
        batch = examples[i : i + batch_size]

        # Build chat-template prompts
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": ex["question"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for ex in batch
        ]

        # Tokenise (left-padded for generation)
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg_global.max_prompt_length,
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,        # greedy for deterministic eval
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens (strip the input prompt)
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[-1]
            new_tokens = output[input_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(text.strip())

    return responses


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_examples(test_file: str) -> list[dict]:
    """Load test.jsonl and return a list of {question, reference} dicts."""
    path = Path(test_file)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")

    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj["messages"]
            # Format: [system, user, assistant]
            examples.append({
                "question":  msgs[1]["content"],   # user turn
                "reference": msgs[2]["content"],   # assistant turn
            })

    print(f"Loaded {len(examples)} test examples from {path}")
    return examples


# ── CSV output ────────────────────────────────────────────────────────────────

def save_results(rows: list[dict], output_path: str) -> None:
    """Write results to CSV with a MEAN summary row appended."""
    df = pd.DataFrame(rows)

    # Numeric columns to average
    metric_cols = [
        "factual_accuracy",
        "relevance",
        "conciseness",
        "no_hallucination",
        "composite_score",
    ]

    # Build mean row (only numeric columns)
    mean_vals = {col: df[col].mean() for col in metric_cols if col in df.columns}
    mean_row = {
        "idx": "MEAN",
        "question": "",
        "reference": "",
        "generated_response": "",
        **mean_vals,
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nResults saved to: {out}")


def print_summary(rows: list[dict]) -> None:
    """Print aggregate metrics to stdout."""
    metric_cols = [
        "factual_accuracy",
        "relevance",
        "conciseness",
        "no_hallucination",
        "composite_score",
    ]
    df = pd.DataFrame(rows)
    print("\n" + "=" * 50)
    print("  Evaluation Summary")
    print("=" * 50)
    for col in metric_cols:
        if col in df.columns:
            scale = "(0–1)" if col == "composite_score" else "(1–5)"
            print(f"  {col:<22} {df[col].mean():.4f}  {scale}")
    print("=" * 50)
    print(f"  n = {len(df)} examples")
    print("=" * 50)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Gemma 3 12B checkpoint on the test set."
    )
    parser.add_argument(
        "--checkpoint",
        default="fine-tuning/gemma3-12b-grpo/checkpoints/final",
        help=(
            "Path to the LoRA adapter directory. "
            "Pass 'none' to evaluate the base model without any adapter. "
            "(default: fine-tuning/gemma3-12b-grpo/checkpoints/final)"
        ),
    )
    parser.add_argument(
        "--test-file",
        default="data/test.jsonl",
        help="Path to test JSONL file. (default: data/test.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="fine-tuning/gemma3-12b-grpo/eval_results.csv",
        help="Output CSV path. (default: fine-tuning/gemma3-12b-grpo/eval_results.csv)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per response. (default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size. Reduce if OOM. (default: 4)",
    )
    parser.add_argument(
        "--no-geval",
        action="store_true",
        help="Skip G-Eval scoring — saves OpenAI cost; useful for generation-only checks.",
    )
    parser.add_argument(
        "--geval-workers",
        type=int,
        default=16,
        help="Parallel threads for G-Eval API calls. (default: 16)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Truncate test set to this many examples (useful for quick checks).",
    )
    return parser.parse_args()


# ── Global config reference (used inside generate_responses) ─────────────────
cfg_global: Config = Config()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global cfg_global
    args = parse_args()

    cfg = Config()
    cfg.geval_max_workers = args.geval_workers
    cfg_global = cfg  # expose to generate_responses

    # ── Validate env vars ──────────────────────────────────────────────────────
    missing = ["HF_TOKEN"] + (
        [] if args.no_geval else ["OPENAI_API_KEY"]
    )
    missing = [v for v in missing if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    # ── Load test data ─────────────────────────────────────────────────────────
    examples = load_test_examples(args.test_file)
    if args.max_samples is not None:
        examples = examples[: args.max_samples]
        print(f"Truncated to {len(examples)} examples (--max-samples).")

    # ── Load model ─────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, cfg)

    # ── Generate responses ─────────────────────────────────────────────────────
    responses = generate_responses(
        examples, model, tokenizer,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    # ── Assemble rows ──────────────────────────────────────────────────────────
    rows: list[dict] = []
    for i, (ex, resp) in enumerate(zip(examples, responses)):
        rows.append({
            "idx":                i,
            "question":           ex["question"],
            "reference":          ex["reference"],
            "generated_response": resp,
        })

    # ── G-Eval scoring ─────────────────────────────────────────────────────────
    if not args.no_geval:
        scores = score_all_parallel(rows, cfg)
        for row, score in zip(rows, scores):
            row.update(score)
    else:
        print("\nSkipping G-Eval (--no-geval). Metric columns will be absent from CSV.")

    # ── Save + summarise ───────────────────────────────────────────────────────
    save_results(rows, args.output)
    if not args.no_geval:
        print_summary(rows)


if __name__ == "__main__":
    main()
