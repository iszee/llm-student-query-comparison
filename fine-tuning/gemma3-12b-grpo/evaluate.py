"""
evaluate.py
-----------
Evaluate Gemma 3 12B across all 8 configurations by default:

    base model        × { plain | sysprompt | fewshot | sysprompt_fewshot }
    fine-tuned model  × { plain | sysprompt | fewshot | sysprompt_fewshot }

Saves one CSV per config and prints a side-by-side comparison table.
Each model is loaded once and reused for all 4 of its prompt variants.

Usage:
    # Full 8-config run (default)
    python fine-tuning/gemma3-12b-grpo/evaluate.py

    # Specific checkpoint
    python fine-tuning/gemma3-12b-grpo/evaluate.py \\
        --checkpoint fine-tuning/gemma3-12b-grpo/checkpoints/checkpoint-100

    # Skip fine-tuned model — base model only (4 configs)
    python fine-tuning/gemma3-12b-grpo/evaluate.py --checkpoint none

    # Dry run — no G-Eval cost
    python fine-tuning/gemma3-12b-grpo/evaluate.py --no-geval

Requires env vars:
    HF_TOKEN        — gated Gemma 3 weights on HuggingFace Hub
    OPENAI_API_KEY  — G-Eval scoring via OpenAI (not needed with --no-geval)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from reward import GEVAL_TEMPLATE, _get_client

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful information assistant for the UQ Bachelor of Information "
    "Technology program. Answer student questions accurately and concisely. "
    "Only provide information you are confident about. "
    "If you are not certain about a specific fact, say so and direct the student "
    "to study.uq.edu.au rather than guessing."
)

METRIC_COLS = [
    "factual_accuracy",
    "relevance",
    "conciseness",
    "no_hallucination",
    "composite_score",
]


# ── Prompt variant ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PromptVariant:
    use_system_prompt: bool
    use_few_shot: bool

    @property
    def label(self) -> str:
        """Short snake_case label used in filenames and table rows."""
        if not self.use_system_prompt and not self.use_few_shot:
            return "plain"
        parts = []
        if self.use_system_prompt:
            parts.append("sysprompt")
        if self.use_few_shot:
            parts.append("fewshot")
        return "_".join(parts)

    @property
    def display(self) -> str:
        """Human-readable label for the comparison table."""
        parts = []
        if self.use_system_prompt:
            parts.append("system prompt")
        if self.use_few_shot:
            parts.append("few-shot")
        return " + ".join(parts) if parts else "plain"


ALL_VARIANTS: list[PromptVariant] = [
    PromptVariant(use_system_prompt=False, use_few_shot=False),
    PromptVariant(use_system_prompt=True,  use_few_shot=False),
    PromptVariant(use_system_prompt=False, use_few_shot=True),
    PromptVariant(use_system_prompt=True,  use_few_shot=True),
]


# ── Few-shot loader ───────────────────────────────────────────────────────────

def load_few_shot_turns(path: str) -> list[dict]:
    """
    Load few-shot example turns from few_shot_examples.json.
    Returns just the 'turns' list (alternating user/assistant messages).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Few-shot file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    turns = data.get("turns", [])
    if not turns:
        raise ValueError(f"'turns' key missing or empty in {p}")
    return turns


def build_messages(
    question: str,
    variant: PromptVariant,
    few_shot_turns: list[dict],
) -> list[dict]:
    """Build the messages list for a single question given a prompt variant."""
    messages: list[dict] = []

    if variant.use_system_prompt:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    if variant.use_few_shot:
        messages.extend(few_shot_turns)

    messages.append({"role": "user", "content": question})
    return messages


# ── G-Eval scoring ────────────────────────────────────────────────────────────

def score_detailed(
    question: str,
    reference: str,
    prediction: str,
    cfg: Config,
) -> dict[str, float]:
    """
    Score one completion with G-Eval (GPT-4o-mini).
    Returns per-dimension scores (1–5) and a composite score (0–1).
    Falls back to zeros after all retries fail.
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
            dims: dict[str, float] = {}
            weighted = 0.0
            for dim, weight in cfg.geval_weights.items():
                val = max(1.0, min(5.0, float(raw.get(dim, 1))))
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
    return {d: 1.0 for d in cfg.geval_weights} | {"composite_score": 0.0}


def score_all_parallel(rows: list[dict], cfg: Config) -> list[dict[str, float]]:
    """Score a list of {question, reference, generated_response} rows in parallel."""
    results: list[dict[str, float]] = [{}] * len(rows)
    with ThreadPoolExecutor(max_workers=cfg.geval_max_workers) as pool:
        futures = {
            pool.submit(
                score_detailed,
                r["question"], r["reference"], r["generated_response"], cfg,
            ): i
            for i, r in enumerate(rows)
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="G-Eval scoring", unit="example"):
            results[futures[future]] = future.result()
    return results


# ── Model loading / unloading ─────────────────────────────────────────────────

def load_model_and_tokenizer(checkpoint: str | None, cfg: Config):
    """
    Load base Gemma 3 12B in BF16 + optional LoRA adapter.

    Args:
        checkpoint: path to a PEFT adapter directory, "none"/None for base model only.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import huggingface_hub

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        huggingface_hub.login(token=hf_token)

    print(f"  Loading base model: {cfg.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if checkpoint and checkpoint.lower() != "none":
        adapter_path = Path(checkpoint)
        if not adapter_path.exists():
            print(f"  WARNING: Checkpoint not found: {adapter_path} — running base model.")
        else:
            print(f"  Loading LoRA adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model.eval()
    else:
        print("  No adapter — evaluating base model.")

    return model, tokenizer


def unload_model(model) -> None:
    """Free GPU memory before loading the next model."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Generation ────────────────────────────────────────────────────────────────

def generate_responses(
    examples: list[dict],
    model,
    tokenizer,
    cfg: Config,
    max_new_tokens: int,
    batch_size: int,
    variant: PromptVariant,
    few_shot_turns: list[dict],
) -> list[str]:
    """Generate one response per example using the given prompt variant."""
    responses: list[str] = []

    for i in tqdm(
        range(0, len(examples), batch_size),
        desc=f"Generating [{variant.display}]",
        unit="batch",
    ):
        batch = examples[i : i + batch_size]

        prompts = [
            tokenizer.apply_chat_template(
                build_messages(ex["question"], variant, few_shot_turns),
                tokenize=False,
                add_generation_prompt=True,
            )
            for ex in batch
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_prompt_length,
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[-1]
            text = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            responses.append(text.strip())

    return responses


# ── Per-variant evaluation ────────────────────────────────────────────────────

def evaluate_variant(
    model,
    tokenizer,
    examples: list[dict],
    cfg: Config,
    max_new_tokens: int,
    batch_size: int,
    run_geval: bool,
    variant: PromptVariant,
    few_shot_turns: list[dict],
) -> list[dict]:
    """Run generation (+ optional G-Eval) for one model × one prompt variant."""
    responses = generate_responses(
        examples, model, tokenizer, cfg,
        max_new_tokens, batch_size, variant, few_shot_turns,
    )

    rows: list[dict] = []
    for i, (ex, resp) in enumerate(zip(examples, responses)):
        rows.append({
            "idx":                i,
            "question":           ex["question"],
            "reference":          ex["reference"],
            "generated_response": resp,
        })

    if run_geval:
        for row, score in zip(rows, score_all_parallel(rows, cfg)):
            row.update(score)

    return rows


def evaluate_all_variants(
    model,
    tokenizer,
    examples: list[dict],
    cfg: Config,
    max_new_tokens: int,
    batch_size: int,
    run_geval: bool,
    few_shot_turns: list[dict],
) -> dict[str, list[dict]]:
    """Run all 4 prompt variants for one loaded model. Returns {label: rows}."""
    return {
        v.label: evaluate_variant(
            model, tokenizer, examples, cfg,
            max_new_tokens, batch_size, run_geval, v, few_shot_turns,
        )
        for v in ALL_VARIANTS
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_examples(test_file: str) -> list[dict]:
    """Load test.jsonl → list of {question, reference} dicts."""
    path = Path(test_file)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line)["messages"]
            examples.append({
                "question":  msgs[1]["content"],
                "reference": msgs[2]["content"],
            })
    print(f"Loaded {len(examples)} test examples from {path}")
    return examples


# ── Output helpers ────────────────────────────────────────────────────────────

def save_results(rows: list[dict], output_path: str) -> None:
    """Write rows to CSV with a MEAN summary row appended."""
    df = pd.DataFrame(rows)
    present = [c for c in METRIC_COLS if c in df.columns]
    mean_row = {
        "idx": "MEAN", "question": "", "reference": "", "generated_response": "",
        **{c: df[c].mean() for c in present},
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  Saved → {out}")


def _mean(rows: list[dict], col: str) -> float | None:
    vals = [r[col] for r in rows if col in r]
    return sum(vals) / len(vals) if vals else None


def print_single_summary(rows: list[dict], label: str = "") -> None:
    """Print aggregate metrics for one run."""
    width = 50
    header = f"  Evaluation Summary{' — ' + label if label else ''}"
    print("\n" + "═" * width)
    print(header)
    print("═" * width)
    for col in METRIC_COLS:
        m = _mean(rows, col)
        if m is not None:
            scale = "(0–1)" if col == "composite_score" else "(1–5)"
            print(f"  {col:<22} {m:.4f}  {scale}")
    print("═" * width)
    print(f"  n = {len(rows)}")
    print("═" * width)


def print_ablation_table(
    results: dict[str, dict[str, list[dict]]],
) -> None:
    """
    Print a full comparison table across all model × prompt-variant combinations.

    Args:
        results: {model_label: {variant_label: rows}}
    """
    col_w = 12      # width of each score column
    lbl_w = 22      # width of the row-label column

    # Header rows
    model_labels   = list(results.keys())
    variant_labels = [v.label   for v in ALL_VARIANTS]
    variant_names  = [v.display for v in ALL_VARIANTS]

    # ── composite_score summary (most important) ──────────────────────────────
    print("\n" + "═" * 80)
    print("  Ablation — composite_score (0–1)")
    print("═" * 80)
    header = f"  {'Model / Variant':<{lbl_w}}"
    for vn in variant_names:
        header += f"  {vn:>{col_w}}"
    print(header)
    print("─" * 80)
    for mlabel in model_labels:
        row = f"  {mlabel:<{lbl_w}}"
        for vlabel in variant_labels:
            rows = results[mlabel].get(vlabel, [])
            m = _mean(rows, "composite_score")
            row += f"  {m:>{col_w}.4f}" if m is not None else f"  {'n/a':>{col_w}}"
        print(row)
    print("═" * 80)

    # ── full metrics per model × variant ─────────────────────────────────────
    for mlabel in model_labels:
        print(f"\n  {'─' * 60}")
        print(f"  Model: {mlabel}")
        print(f"  {'─' * 60}")
        print(f"  {'Metric':<22}  " + "  ".join(f"{v.display:>{col_w}}" for v in ALL_VARIANTS))
        print(f"  {'─' * 60}")
        for col in METRIC_COLS:
            scale = "(0–1)" if col == "composite_score" else "(1–5)"
            row = f"  {col:<22}"
            for vlabel in variant_labels:
                rows = results[mlabel].get(vlabel, [])
                m = _mean(rows, col)
                row += f"  {m:>{col_w}.4f}" if m is not None else f"  {'n/a':>{col_w}}"
            print(row + f"  {scale}")
        print(f"  {'─' * 60}")
        print(f"  n = {len(next(iter(results[mlabel].values())))} examples")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Gemma 3 12B across all 8 configurations: "
            "base + fine-tuned × 4 prompt variants "
            "(plain / sysprompt / fewshot / sysprompt_fewshot)."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="fine-tuning/gemma3-12b-grpo/checkpoints/final",
        help=(
            "Path to the LoRA adapter directory. "
            "Pass 'none' to skip the fine-tuned model and run base only. "
            "(default: fine-tuning/gemma3-12b-grpo/checkpoints/final)"
        ),
    )
    parser.add_argument(
        "--test-file",
        default="data/test.jsonl",
        help="Path to test JSONL file. (default: data/test.jsonl)",
    )
    parser.add_argument(
        "--few-shot-file",
        default="data/few_shot_examples.json",
        help="Path to few-shot examples JSON. (default: data/few_shot_examples.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="fine-tuning/gemma3-12b-grpo/results",
        help=(
            "Directory for output CSVs. Files are named eval_{model}_{variant}.csv. "
            "(default: fine-tuning/gemma3-12b-grpo/results)"
        ),
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
        help="Skip G-Eval scoring — no OpenAI cost; useful for generation-only checks.",
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
        help="Truncate test set to N examples (useful for quick checks).",
    )
    return parser.parse_args()


# ── Summary CSV ───────────────────────────────────────────────────────────────

def save_summary_csv(
    all_results: dict[str, dict[str, list[dict]]],
    output_dir: str,
) -> None:
    """
    Save one row per model × variant with mean metric values to eval_summary.csv.

    Columns: model, variant, factual_accuracy, relevance, conciseness,
             no_hallucination, composite_score
    """
    rows = []
    for model_label, variant_results in all_results.items():
        for vlabel, result_rows in variant_results.items():
            row: dict = {"model": model_label, "variant": vlabel}
            for col in METRIC_COLS:
                row[col] = _mean(result_rows, col)   # None if G-Eval was skipped
            rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(output_dir) / "eval_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  Summary saved → {out}")


# ── CSV output path helper ────────────────────────────────────────────────────

def result_path(output_dir: str, model_label: str, variant_label: str) -> str:
    return str(Path(output_dir) / f"eval_{model_label}_{variant_label}.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    cfg = Config()
    cfg.geval_max_workers = args.geval_workers
    run_geval = not args.no_geval

    # ── Validate env vars ──────────────────────────────────────────────────────
    required = ["HF_TOKEN"] + ([] if args.no_geval else ["OPENAI_API_KEY"])
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    # ── Load shared data ───────────────────────────────────────────────────────
    examples = load_test_examples(args.test_file)
    if args.max_samples is not None:
        examples = examples[: args.max_samples]
        print(f"Truncated to {len(examples)} examples (--max-samples).")

    few_shot_turns = load_few_shot_turns(args.few_shot_file)

    # ── Models to evaluate ─────────────────────────────────────────────────────
    # Always run base; add fine-tuned unless --checkpoint none
    models_to_run: list[tuple[str, str | None]] = [("base", None)]
    if args.checkpoint.lower() != "none":
        models_to_run.append(("finetuned", args.checkpoint))

    # ── Run all configs ────────────────────────────────────────────────────────
    all_results: dict[str, dict[str, list[dict]]] = {}

    for model_label, checkpoint in models_to_run:
        print(f"\n{'═' * 60}")
        print(f"  Model: {model_label}")
        print(f"{'═' * 60}")

        model, tokenizer = load_model_and_tokenizer(checkpoint, cfg)

        all_results[model_label] = evaluate_all_variants(
            model, tokenizer, examples, cfg,
            args.max_new_tokens, args.batch_size, run_geval, few_shot_turns,
        )

        unload_model(model)

        for vlabel, rows in all_results[model_label].items():
            save_results(rows, result_path(args.output_dir, model_label, vlabel))

    # ── Save summary + print results ──────────────────────────────────────────
    save_summary_csv(all_results, args.output_dir)
    if run_geval:
        print_ablation_table(all_results)
    else:
        print("\nSkipping G-Eval (--no-geval). Metric columns absent from CSVs.")


if __name__ == "__main__":
    main()
