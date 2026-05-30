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

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
    use_rag: bool = False

    @property
    def label(self) -> str:
        """Short snake_case label used in filenames and table rows."""
        if not self.use_system_prompt and not self.use_few_shot and not self.use_rag:
            return "plain"
        parts = []
        if self.use_system_prompt:
            parts.append("sysprompt")
        if self.use_rag:
            parts.append("rag")
        if self.use_few_shot:
            parts.append("fewshot")
        return "_".join(parts)

    @property
    def display(self) -> str:
        """Human-readable label for the comparison table."""
        parts = []
        if self.use_system_prompt:
            parts.append("system prompt")
        if self.use_rag:
            parts.append("RAG")
        if self.use_few_shot:
            parts.append("few-shot")
        return " + ".join(parts) if parts else "plain"


ALL_VARIANTS: list[PromptVariant] = [
    PromptVariant(use_system_prompt=False, use_few_shot=False, use_rag=False),
    PromptVariant(use_system_prompt=True,  use_few_shot=False, use_rag=False),
    PromptVariant(use_system_prompt=False, use_few_shot=True,  use_rag=False),
    PromptVariant(use_system_prompt=True,  use_few_shot=True,  use_rag=False),
    PromptVariant(use_system_prompt=False, use_few_shot=False, use_rag=True),
    PromptVariant(use_system_prompt=True,  use_few_shot=False, use_rag=True),
    PromptVariant(use_system_prompt=False, use_few_shot=True,  use_rag=True),
    PromptVariant(use_system_prompt=True,  use_few_shot=True,  use_rag=True),
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


def _format_reference_block(chunks: list) -> str:
    """Format retrieved chunks as a numbered reference block prepended to the user message."""
    lines = ["Reference material from UQ documents:"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[{i}] ({chunk.display_source}) {chunk.text}")
    lines.append("")
    lines.append(
        "Use the references above to answer accurately. "
        "If the references do not contain the answer, say so."
    )
    return "\n".join(lines)


def build_messages(
    question: str,
    variant: PromptVariant,
    few_shot_turns: list[dict],
    retrieved_chunks: list | None = None,
) -> list[dict]:
    """Build the messages list for a single question given a prompt variant."""
    messages: list[dict] = []

    if variant.use_system_prompt:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    if variant.use_few_shot:
        messages.extend(few_shot_turns)

    if retrieved_chunks:
        user_content = f"{_format_reference_block(retrieved_chunks)}\n\nQuestion: {question}"
    else:
        user_content = question

    messages.append({"role": "user", "content": user_content})
    return messages


# ── G-Eval scoring ────────────────────────────────────────────────────────────

def score_detailed(
    question: str,
    reference: str,
    prediction: str,
    cfg: Config,
) -> dict[str, float]:
    """
    Score one completion with G-Eval (GPT-4o).
    Returns per-dimension scores (0–5 float) and a composite score (0–1).
    Falls back to zeros after all retries fail.
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
                raw = json.loads(resp.choices[0].message.content)
                for dim in cfg.geval_weights:
                    val = max(0.0, min(5.0, float(raw.get(dim, 0))))
                    dim_accum[dim] += val
                successful += 1
                break   # sample succeeded — move to next sample
            except Exception as exc:
                wait = 2 ** attempt
                print(f"[evaluate] G-Eval sample {s+1}/{n_samples} attempt {attempt+1} failed: {exc}. "
                      f"Retrying in {wait}s...")
                time.sleep(wait)

    if successful == 0:
        print("[evaluate] All G-Eval samples exhausted — recording floor scores (dim=0, composite=0).")
        return {d: 0.0 for d in cfg.geval_weights} | {"composite_score": 0.0}

    # Average dims across successful samples, then compute composite
    dims: dict[str, float] = {dim: dim_accum[dim] / successful for dim in cfg.geval_weights}
    weighted = sum(dims[dim] * w for dim, w in cfg.geval_weights.items())
    dims["composite_score"] = weighted / 5.0
    return dims


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
    max_new_tokens: int,
    batch_size: int,
    variant: PromptVariant,
    few_shot_turns: list[dict],
    all_retrieved: list | None = None,
) -> list[str]:
    """Generate one response per example using the given prompt variant."""
    responses: list[str] = []

    for i in tqdm(
        range(0, len(examples), batch_size),
        desc=f"Generating [{variant.display}]",
        unit="batch",
    ):
        batch = examples[i : i + batch_size]
        batch_retrieved = (
            all_retrieved[i : i + batch_size]
            if all_retrieved is not None
            else [None] * len(batch)
        )

        prompts = [
            tokenizer.apply_chat_template(
                build_messages(ex["question"], variant, few_shot_turns, chunks),
                tokenize=False,
                add_generation_prompt=True,
            )
            for ex, chunks in zip(batch, batch_retrieved)
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
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
    retriever=None,
) -> list[dict]:
    """Run generation (+ optional G-Eval) for one model × one prompt variant."""
    all_retrieved = None
    if variant.use_rag and retriever is not None:
        retrieval_cache: dict[str, list] = {}
        all_retrieved = []
        for ex in tqdm(examples, desc=f"Retrieving [{variant.display}]", unit="q"):
            q = ex["question"]
            if q not in retrieval_cache:
                retrieval_cache[q] = retriever.search(q)
            all_retrieved.append(retrieval_cache[q])

    responses = generate_responses(
        examples, model, tokenizer,
        max_new_tokens, batch_size, variant, few_shot_turns,
        all_retrieved=all_retrieved,
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
    retriever=None,
    active_variants: list | None = None,
) -> dict[str, list[dict]]:
    """Run prompt variants for one loaded model. Returns {label: rows}."""
    variants = active_variants if active_variants is not None else ALL_VARIANTS
    return {
        v.label: evaluate_variant(
            model, tokenizer, examples, cfg,
            max_new_tokens, batch_size, run_geval, v, few_shot_turns,
            retriever=retriever,
        )
        for v in variants
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
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"  Saved → {out}")


def _mean(rows: list[dict], col: str) -> float | None:
    vals = [r[col] for r in rows if col in r]
    return sum(vals) / len(vals) if vals else None


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
            scale = "(0–1)" if col == "composite_score" else "(0–5)"
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
        default=None,
        help=(
            "Maximum tokens to generate per response. "
            "Defaults to cfg.max_completion_length (256) to match training."
        ),
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
        "--no-rag",
        action="store_true",
        help="Skip all RAG variants — no retrieval cost. Runs only the 4 non-RAG configs.",
    )
    parser.add_argument(
        "--index-dir",
        default="rag/index",
        help="Path to the RAG index directory. (default: rag/index)",
    )
    parser.add_argument(
        "--geval-workers",
        type=int,
        default=16,
        help="Parallel threads for G-Eval API calls. (default: 16)",
    )
    parser.add_argument(
        "--geval-samples",
        type=int,
        default=None,
        help=(
            "Number of judge samples to average per example (G-Eval paper). "
            "1 = single deterministic call; ≥2 = sampling at --geval-sample-temp. "
            "Defaults to cfg.geval_samples (3)."
        ),
    )
    parser.add_argument(
        "--geval-sample-temp",
        type=float,
        default=None,
        help="Sampling temperature when --geval-samples > 1. Defaults to cfg.geval_sample_temperature (0.5).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Truncate test set to N examples (useful for quick checks).",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help=(
            "Re-score existing result CSVs without loading any model. "
            "Reads eval_base_<v>.csv and eval_finetuned_<v>.csv from --output-dir, "
            "rescores with the improved judge, and writes eval_summary_v2.csv."
        ),
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
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"  Summary saved → {out}")


# ── CSV output path helper ────────────────────────────────────────────────────

def result_path(output_dir: str, model_label: str, variant_label: str) -> str:
    return str(Path(output_dir) / f"eval_{model_label}_{variant_label}.csv")


# ── Rescore mode ──────────────────────────────────────────────────────────────

def rescore_from_csvs(args, cfg: Config) -> None:
    """
    Re-score existing result CSVs without loading any model.

    Reads eval_base_<v>.csv and eval_finetuned_<v>.csv from --output-dir,
    rescores each generated_response with the improved judge (float 0–5, reasoning),
    and writes eval_summary_v2.csv alongside the originals.
    Run with: python evaluate.py --rescore [--geval-workers N]
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is required for --rescore.")
        sys.exit(1)

    output_dir = args.output_dir
    all_results: dict[str, dict[str, list[dict]]] = {}

    for model_label in ("base", "finetuned"):
        model_results: dict[str, list[dict]] = {}
        for v in ALL_VARIANTS:
            csv_path = result_path(output_dir, model_label, v.label)
            if not Path(csv_path).exists():
                print(f"  Skipping {csv_path} (not found)")
                continue
            df = pd.read_csv(csv_path)
            # Drop the MEAN summary row if present
            rows = df[df["idx"] != "MEAN"].to_dict("records")
            print(f"\n  Rescoring {model_label}/{v.label} ({len(rows)} examples)…")
            for row, new_scores in zip(rows, score_all_parallel(rows, cfg)):
                row.update(new_scores)
            # Save per-variant v2 CSV
            out_path = csv_path.replace(".csv", "_v2.csv")
            save_results(rows, out_path)
            model_results[v.label] = rows
        all_results[model_label] = model_results

    # Save v2 summary
    rows_summary = []
    for model_label, variant_results in all_results.items():
        for vlabel, result_rows in variant_results.items():
            row: dict = {"model": model_label, "variant": vlabel}
            for col in METRIC_COLS:
                row[col] = _mean(result_rows, col)
            rows_summary.append(row)

    df_summary = pd.DataFrame(rows_summary)
    out = Path(output_dir) / "eval_summary_v2.csv"
    df_summary.to_csv(out, index=False, encoding="utf-8")
    print(f"\n  Summary saved → {out}")
    print_ablation_table(all_results)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    cfg = Config()
    cfg.geval_max_workers = args.geval_workers
    if getattr(args, "geval_samples", None) is not None:
        cfg.geval_samples = args.geval_samples
    if getattr(args, "geval_sample_temp", None) is not None:
        cfg.geval_sample_temperature = args.geval_sample_temp

    # ── Rescore mode: re-score existing CSVs, no model load ───────────────────
    if getattr(args, "rescore", False):
        rescore_from_csvs(args, cfg)
        return

    run_geval = not args.no_geval

    # Default max_new_tokens to match training's max_completion_length
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else cfg.max_completion_length

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

    # ── RAG retriever ──────────────────────────────────────────────────────────
    active_variants = [v for v in ALL_VARIANTS if not (v.use_rag and args.no_rag)]
    retriever = None
    if any(v.use_rag for v in active_variants):
        try:
            from rag import Retriever
            retriever = Retriever(args.index_dir)
            print(f"RAG index loaded from {args.index_dir}.")
        except ImportError:
            print(
                "WARNING: RAG dependencies not installed. "
                "Run: pip install sentence-transformers faiss-cpu rank_bm25\n"
                "Skipping RAG variants for this run."
            )
            active_variants = [v for v in active_variants if not v.use_rag]
        except FileNotFoundError:
            print(
                f"WARNING: RAG index not found at '{args.index_dir}'. "
                "Run: python -m rag build\n"
                "Skipping RAG variants for this run."
            )
            active_variants = [v for v in active_variants if not v.use_rag]

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
            max_new_tokens, args.batch_size, run_geval, few_shot_turns,
            retriever=retriever, active_variants=active_variants,
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
