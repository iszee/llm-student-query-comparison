"""
__main__.py
-----------
CLI for the UQ BIT RAG index.

Usage:
    python -m rag build                         # ingest PDFs + URLs, embed, index
    python -m rag build --force                 # force rebuild even if index exists
    python -m rag build --no-contextualise      # skip GPT-4o-mini prefix generation
    python -m rag query "What is the ATAR?"     # retrieve and print top-5 chunks
    python -m rag query "What is the ATAR?" --top-k 3
"""

import argparse
import sys
from pathlib import Path


def cmd_build(args: argparse.Namespace) -> None:
    from rag.build_index import build
    build(force=args.force, contextualise=not args.no_contextualise)


def cmd_query(args: argparse.Namespace) -> None:
    from rag.retrieve import Retriever

    retriever = Retriever(args.index_dir)
    results = retriever.search(
        args.question,
        top_k=args.top_k,
        min_score=args.min_score,
        llm_filter=not args.no_llm_filter,
    )

    if not results:
        print("No results found.")
        return

    print(f"\nTop {len(results)} results for: {args.question!r}")
    print("─" * 70)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r.display_source}  (score: {r.score:.4f})")
        if r.context_prefix:
            print(f"    Context: {r.context_prefix}")
        snippet = r.text[:300]
        if len(r.text) > 300:
            snippet += "..."
        print(f"    {snippet}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m rag",
        description="RAG index management for the UQ BIT information assistant.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Ingest sources and build the retrieval index.")
    p_build.add_argument(
        "--force", action="store_true",
        help="Rebuild even if the index already exists.",
    )
    p_build.add_argument(
        "--no-contextualise", action="store_true",
        help="Skip GPT-4o-mini context-prefix generation (no OpenAI cost, lower quality).",
    )
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser("query", help="Query the index and print top-k retrieved chunks.")
    p_query.add_argument("question", help="Question to retrieve context for.")
    p_query.add_argument(
        "--top-k", type=int, default=5,
        help="Maximum chunks to return after filtering. (default: 5)",
    )
    p_query.add_argument(
        "--min-score", type=float, default=0.30,
        help="Minimum cross-encoder rerank score to keep a chunk (0–1). "
             "Chunks below this are dropped; returns [] if none survive. (default: 0.30)",
    )
    p_query.add_argument(
        "--no-llm-filter", action="store_true",
        help="Disable the gpt-4o-mini relevance filter (score threshold still applies).",
    )
    p_query.add_argument(
        "--index-dir", default="rag/index",
        help="Path to the RAG index directory. (default: rag/index)",
    )
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
