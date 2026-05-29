"""
retrieve.py
-----------
Hybrid RAG retriever for the UQ BIT information assistant.

Pipeline: dense (BGE-M3) + BM25 → Reciprocal Rank Fusion → cross-encoder rerank.

References:
  Embeddings : Chen et al. 2024, "BGE M3-Embedding" — arXiv:2402.03216
  Fusion     : Cormack, Clarke, Büttcher 2009, "Reciprocal Rank Fusion" — SIGIR 2009
  Reranking  : Li et al. 2023, "bge-reranker-v2-m3" — arXiv:2312.15503
"""

from __future__ import annotations

import json
import pickle
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

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """
        Full pipeline: embed → dense FAISS → BM25 → RRF fusion → cross-encoder rerank.
        Returns top_k RetrievedChunk objects ordered by rerank score (descending).
        """
        n_candidates = max(top_k * 4, 20)

        dense_hits = self._dense_search(query, n_candidates)
        bm25_hits  = self._bm25_search(query, n_candidates)
        fused_ids  = self._rrf(dense_hits, bm25_hits, n_candidates)
        return self._rerank(query, fused_ids, top_k)

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

    def _rerank(self, query: str, candidate_ids: list[int], top_k: int) -> list[RetrievedChunk]:
        """Cross-encoder rerank with bge-reranker-v2-m3 (Li et al. 2023)."""
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
        for chunk_id, score in ranked[:top_k]:
            c = self._chunks[chunk_id]
            results.append(RetrievedChunk(
                text=c["text"],
                context_prefix=c.get("context_prefix", ""),
                source=c["source"],
                page=c.get("page"),
                score=float(score),
            ))
        return results
