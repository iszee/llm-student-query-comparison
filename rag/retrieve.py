"""
retrieve.py
-----------
Hybrid RAG retriever for the UQ BIT information assistant.

Pipeline: dense (BGE-M3) + BM25 → Reciprocal Rank Fusion → cross-encoder rerank
         → score threshold → LLM relevance filter (gpt-4o-mini).

References:
  Embeddings : Chen et al. 2024, "BGE M3-Embedding" — arXiv:2402.03216
  Fusion     : Cormack, Clarke, Büttcher 2009, "Reciprocal Rank Fusion" — SIGIR 2009
  Reranking  : Li et al. 2023, "bge-reranker-v2-m3" — arXiv:2312.15503
"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DEFAULT_INDEX_DIR = Path(__file__).parent / "index"
EMBED_MODEL  = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    context_prefix: str
    source: str
    page: int | None
    score: float

    @property
    def display_source(self) -> str:
        """Short label used in the reference block injected into prompts."""
        if self.page is not None:
            return f"{Path(self.source).name}, p.{self.page}"
        return self.source


class Retriever:
    """
    Loads pre-built FAISS + BM25 indices and answers queries via hybrid retrieval + reranking.

    Construct once per evaluation run; `search()` is stateless (read-only) after init.

    Args:
        index_dir: directory containing chunks.jsonl, dense.faiss, bm25.pkl.
                   Defaults to rag/index/ relative to this file.
    """

    def __init__(self, index_dir: str | Path = _DEFAULT_INDEX_DIR) -> None:
        import faiss
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer, CrossEncoder

        self._index_dir = Path(index_dir)
        chunks_path = self._index_dir / "chunks.jsonl"
        faiss_path  = self._index_dir / "dense.faiss"
        bm25_path   = self._index_dir / "bm25.pkl"

        if not chunks_path.exists():
            raise FileNotFoundError(
                f"RAG index not found at {self._index_dir}. "
                "Run `python -m rag build` to create it first."
            )

        self._chunks: list[dict] = []
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._chunks.append(json.loads(line))

        self._faiss = faiss.read_index(str(faiss_path))

        with open(bm25_path, "rb") as f:
            self._bm25: BM25Okapi = pickle.load(f)

        print(f"[rag] Loading embedding model: {EMBED_MODEL}")
        self._embedder = SentenceTransformer(EMBED_MODEL)
        print(f"[rag] Loading reranker: {RERANK_MODEL}")
        self._reranker = CrossEncoder(RERANK_MODEL)
        print(f"[rag] Retriever ready — {len(self._chunks)} chunks indexed.")

    # ── Public API ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.30,
        llm_filter: bool = True,
    ) -> list[RetrievedChunk]:
        """
        Full pipeline: embed → dense FAISS → BM25 → RRF fusion → cross-encoder rerank
                       → score threshold → optional LLM relevance filter → top_k.

        Args:
            query:      The question to retrieve context for.
            top_k:      Maximum chunks to return after all filtering.
            min_score:  Minimum cross-encoder rerank score (0–1) to keep a chunk.
                        Chunks below this threshold are dropped; if none survive,
                        returns [] so callers fall back to unaugmented generation.
                        Default 0.30 — tuned from observed probe data.
            llm_filter: When True, a gpt-4o-mini call checks whether each surviving
                        chunk actually contains information useful for answering the
                        query, filtering out high-scoring but off-topic marketing text
                        (Mode B failure).  Gracefully disabled if OPENAI_API_KEY is
                        unset or the call fails.
        """
        n_candidates = max(top_k * 4, 20)

        dense_hits = self._dense_search(query, n_candidates)
        bm25_hits  = self._bm25_search(query, n_candidates)
        fused_ids  = self._rrf(dense_hits, bm25_hits, n_candidates)
        reranked   = self._rerank(query, fused_ids)  # full sorted list, no cap yet

        # ── Stage 1: score threshold (kills Mode A — corpus has no answer) ──────
        kept = [c for c in reranked if c.score >= min_score]
        if not kept:
            return []

        # ── Stage 2: LLM relevance filter (kills Mode B — marketing fluff) ──────
        if llm_filter:
            kept = self._llm_filter(query, kept)

        return kept[:top_k]

    # ── Private ─────────────────────────────────────────────────────────────────

    def _dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        vec = self._embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self._faiss.search(vec, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def _bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        tokens = query.lower().split()
        raw_scores = self._bm25.get_scores(tokens)
        top_ids = np.argsort(raw_scores)[::-1][:k]
        return [(int(i), float(raw_scores[i])) for i in top_ids if raw_scores[i] > 0]

    def _rrf(
        self,
        dense: list[tuple[int, float]],
        sparse: list[tuple[int, float]],
        k: int,
        rrf_k: int = 60,
    ) -> list[int]:
        """Reciprocal Rank Fusion — Cormack, Clarke, Büttcher (SIGIR 2009)."""
        scores: dict[int, float] = {}
        for rank, (chunk_id, _) in enumerate(dense):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, (chunk_id, _) in enumerate(sparse):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        return [
            cid for cid, _ in
            sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        ]

    def _rerank(self, query: str, candidate_ids: list[int]) -> list[RetrievedChunk]:
        """
        Cross-encoder rerank with bge-reranker-v2-m3 (Li et al. 2023).
        Returns ALL candidates sorted by score descending — caller caps to top_k.
        """
        if not candidate_ids:
            return []

        passages = [
            f"{self._chunks[i]['context_prefix']}\n\n{self._chunks[i]['text']}"
            if self._chunks[i].get("context_prefix")
            else self._chunks[i]["text"]
            for i in candidate_ids
        ]
        pairs = [[query, p] for p in passages]
        re_scores = self._reranker.predict(pairs)

        ranked = sorted(zip(candidate_ids, re_scores), key=lambda x: x[1], reverse=True)

        results: list[RetrievedChunk] = []
        for chunk_id, score in ranked:
            c = self._chunks[chunk_id]
            results.append(RetrievedChunk(
                text=c["text"],
                context_prefix=c.get("context_prefix", ""),
                source=c["source"],
                page=c.get("page"),
                score=float(score),
            ))
        return results

    def _llm_filter(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """
        Use gpt-4o-mini to keep only chunks that genuinely help answer the query.

        One API call batches all candidates so cost is one request per unique query
        (retrieval is cached per-question in evaluate.py, so this is called at most
        once per question per run).

        Gracefully falls back to returning `chunks` unfiltered when:
          - OPENAI_API_KEY is not set
          - the API call fails (after retries)
          - the response cannot be parsed as valid JSON
        Retrieval must never hard-fail.
        """
        if not os.environ.get("OPENAI_API_KEY"):
            print("[rag] OPENAI_API_KEY not set — skipping LLM relevance filter.")
            return chunks

        # Build numbered candidate list using context_prefix for cheap signal
        lines = [
            f"[{i}] ({c.context_prefix.strip()})\n    {c.text[:300]}"
            if c.context_prefix
            else f"[{i}] {c.text[:300]}"
            for i, c in enumerate(chunks)
        ]
        candidate_block = "\n\n".join(lines)

        prompt = (
            "You are a relevance filter for a retrieval-augmented QA system.\n"
            "Given a question and a numbered list of retrieved passages, return ONLY\n"
            "a JSON object {\"relevant\": [list of integer indices]} for the passages\n"
            "that contain information DIRECTLY useful for answering the question.\n"
            "Exclude passages that are generic, promotional, or unrelated.\n\n"
            f"Question: {query}\n\n"
            f"Passages:\n{candidate_block}"
        )

        from openai import OpenAI
        client = OpenAI()

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=64,
                    response_format={"type": "json_object"},
                    timeout=30,
                )
                data = json.loads(resp.choices[0].message.content)
                relevant_ids = [int(i) for i in data.get("relevant", [])]
                kept = [chunks[i] for i in relevant_ids if 0 <= i < len(chunks)]
                # Preserve rerank order (already sorted by score desc)
                kept.sort(key=lambda c: chunks.index(c))
                return kept if kept else chunks  # never return empty from filter
            except Exception as exc:
                wait = 2 ** attempt
                print(f"[rag] LLM filter attempt {attempt + 1} failed: {exc}. "
                      f"Retrying in {wait}s…")
                time.sleep(wait)

        print("[rag] LLM filter failed after 3 attempts — returning chunks unfiltered.")
        return chunks
