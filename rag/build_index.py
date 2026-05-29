"""
build_index.py
--------------
One-shot ingestion pipeline for the UQ BIT RAG index.

Sources:
  - All PDFs in data-collection/sources/*.pdf   (drop files there to include them)
  - URLs from data-collection/sources/program-pages.txt

Pipeline:
  1. Extract text (pypdf for PDFs, requests+bs4 for URLs)
  2. Chunk — 512-token windows with 64-token overlap
  3. Contextualise — GPT-4o-mini writes a 1-sentence context prefix per chunk
     (Anthropic 2024, "Introducing Contextual Retrieval" — 49% fewer retrieval failures)
  4. Embed — BAAI/bge-m3 dense embeddings (Chen et al. 2024, arXiv:2402.03216)
  5. BM25 — rank_bm25 sparse index
  6. Persist — rag/index/{chunks.jsonl, dense.faiss, bm25.pkl}

Usage:
    python -m rag build                      # full pipeline
    python -m rag build --force              # rebuild even if index exists
    python -m rag build --no-contextualise   # skip GPT-4o-mini (no cost, lower quality)
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

SOURCES_DIR        = Path("data-collection/sources")
PROGRAM_PAGES_FILE = SOURCES_DIR / "program-pages.txt"
INDEX_DIR          = Path("rag/index")
CHUNKS_FILE        = INDEX_DIR / "chunks.jsonl"
FAISS_FILE         = INDEX_DIR / "dense.faiss"
BM25_FILE          = INDEX_DIR / "bm25.pkl"
CTX_CACHE_FILE     = INDEX_DIR / "context_cache.json"

EMBED_MODEL  = "BAAI/bge-m3"
CTX_MODEL    = "gpt-4o-mini"
CHUNK_WORDS  = 512    # target chunk size in whitespace tokens (≈ words)
OVERLAP_WORDS = 64    # overlap between adjacent chunks

_CONTEXT_SYSTEM = (
    "You are a document analyst. Given a large document and a short excerpt, "
    "write a single sentence (max 20 words) describing what section or topic "
    "the excerpt is from — so a reader can immediately place it in context. "
    "Output only the sentence, no preamble."
)
_CONTEXT_USER = (
    "Document excerpt (for context only, first ~600 words):\n{doc_excerpt}\n\n"
    "Chunk to contextualise:\n{chunk}\n\n"
    "Write one sentence describing where this chunk sits in the document."
)


# ── Chunking ───────────────────────────────────────────────────────────────────

def _chunk_text(text: str, source: str, page: int | None = None) -> list[dict]:
    """
    Whitespace-token sliding window: CHUNK_WORDS tokens with OVERLAP_WORDS overlap.
    Returns list of {source, page, text} dicts; skips near-empty chunks.
    """
    words = text.split()
    chunks: list[dict] = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_WORDS, len(words))
        chunk_text = " ".join(words[start:end]).strip()
        if len(chunk_text) > 60:
            chunks.append({"source": source, "page": page, "text": chunk_text})
        if end >= len(words):
            break
        start += CHUNK_WORDS - OVERLAP_WORDS
    return chunks


# ── PDF ingest ─────────────────────────────────────────────────────────────────

def ingest_pdfs(sources_dir: Path) -> list[dict]:
    from pypdf import PdfReader

    chunks: list[dict] = []
    pdfs = sorted(sources_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[build] No PDFs found in {sources_dir}")
        return chunks

    print(f"[build] Found {len(pdfs)} PDF(s): {[p.name for p in pdfs]}")
    for pdf_path in pdfs:
        print(f"  Parsing {pdf_path.name} ...")
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            text = re.sub(r"\s+", " ", raw).strip()
            if len(text) < 30:
                continue
            chunks.extend(_chunk_text(text, source=pdf_path.name, page=page_num))

    print(f"  → {len(chunks)} chunks from PDFs")
    return chunks


# ── URL ingest ─────────────────────────────────────────────────────────────────

def _parse_urls(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _fetch_url(url: str) -> str:
    import requests
    from bs4 import BeautifulSoup

    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as exc:
        print(f"  WARNING: Failed to fetch {url}: {exc}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()


def ingest_urls(pages_file: Path) -> list[dict]:
    if not pages_file.exists():
        print(f"[build] {pages_file} not found — skipping URL ingest.")
        return []

    urls = _parse_urls(pages_file)
    print(f"[build] Fetching {len(urls)} program page URLs ...")
    chunks: list[dict] = []
    for url in urls:
        print(f"  {url}")
        text = _fetch_url(url)
        if text:
            chunks.extend(_chunk_text(text, source=url, page=None))
        time.sleep(0.5)

    print(f"  → {len(chunks)} chunks from URLs")
    return chunks


# ── Contextual Retrieval (Anthropic 2024) ──────────────────────────────────────

def _get_openai_client():
    from openai import OpenAI
    return OpenAI()


def _generate_prefix(client, doc_excerpt: str, chunk_text: str) -> str:
    """Call GPT-4o-mini to write a 1-sentence context prefix for one chunk."""
    try:
        resp = client.chat.completions.create(
            model=CTX_MODEL,
            messages=[
                {"role": "system", "content": _CONTEXT_SYSTEM},
                {"role": "user", "content": _CONTEXT_USER.format(
                    doc_excerpt=doc_excerpt[:3000],
                    chunk=chunk_text[:800],
                )},
            ],
            max_tokens=60,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        print(f"  WARNING: context prefix generation failed: {exc}")
        return ""


def add_context_prefixes(chunks: list[dict], cache_file: Path) -> list[dict]:
    """
    Add 'context_prefix' to each chunk using GPT-4o-mini.

    Implements Anthropic's Contextual Retrieval technique (2024):
      https://www.anthropic.com/news/contextual-retrieval
    Results are cached by chunk-text hash so rebuilds don't re-pay.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print("[build] OPENAI_API_KEY not set — skipping contextualisation (empty prefixes).")
        for c in chunks:
            c["context_prefix"] = ""
        return chunks

    cache: dict[str, str] = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
            print(f"[build] Loaded {len(cache)} cached context prefixes.")
        except Exception:
            pass

    client = _get_openai_client()

    # Group by source to build a per-source document excerpt
    by_source: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chunks):
        by_source[c["source"]].append(i)

    cached_count = sum(1 for c in chunks if str(hash(c["text"])) in cache)
    total = len(chunks)
    print(f"[build] Generating context prefixes: {total - cached_count} new / {total} total ...")

    for source, idxs in by_source.items():
        doc_excerpt = " ".join(chunks[i]["text"] for i in idxs[:6])  # first ~3000 words as context

        for idx in idxs:
            chunk_hash = str(hash(chunks[idx]["text"]))
            if chunk_hash in cache:
                chunks[idx]["context_prefix"] = cache[chunk_hash]
                continue

            prefix = _generate_prefix(client, doc_excerpt, chunks[idx]["text"])
            chunks[idx]["context_prefix"] = prefix
            cache[chunk_hash] = prefix

            # Save incrementally so partial runs aren't lost
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    print(f"[build] Context prefix generation done. Cache: {cache_file}")
    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def build_faiss_index(chunks: list[dict]):
    import faiss
    from sentence_transformers import SentenceTransformer

    print(f"[build] Encoding {len(chunks)} chunks with {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [
        f"{c['context_prefix']}\n\n{c['text']}" if c.get("context_prefix") else c["text"]
        for c in chunks
    ]
    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[build] FAISS index: {index.ntotal} vectors, dim={dim}")
    return index


def build_bm25_index(chunks: list[dict]):
    from rank_bm25 import BM25Okapi

    corpus = [
        (
            f"{c['context_prefix']} {c['text']}" if c.get("context_prefix") else c["text"]
        ).lower().split()
        for c in chunks
    ]
    return BM25Okapi(corpus)


# ── Persist ────────────────────────────────────────────────────────────────────

def persist(chunks: list[dict], faiss_index, bm25_index, index_dir: Path) -> None:
    import faiss

    index_dir.mkdir(parents=True, exist_ok=True)

    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            f.write(json.dumps({"id": i, **c}, ensure_ascii=False) + "\n")

    faiss.write_index(faiss_index, str(index_dir / "dense.faiss"))

    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)

    print(f"[build] Index persisted to {index_dir}/  ({len(chunks)} chunks)")


# ── Entry point ────────────────────────────────────────────────────────────────

def build(force: bool = False, contextualise: bool = True) -> None:
    if CHUNKS_FILE.exists() and not force:
        print(
            f"[build] Index already exists at {INDEX_DIR}. "
            "Pass --force to rebuild."
        )
        return

    chunks = ingest_pdfs(SOURCES_DIR)
    chunks += ingest_urls(PROGRAM_PAGES_FILE)

    if not chunks:
        raise RuntimeError(
            "No content ingested. "
            "Ensure data-collection/sources/*.pdf or program-pages.txt URLs are available."
        )

    if contextualise:
        chunks = add_context_prefixes(chunks, CTX_CACHE_FILE)
    else:
        for c in chunks:
            c["context_prefix"] = ""

    faiss_index = build_faiss_index(chunks)
    bm25_index  = build_bm25_index(chunks)
    persist(chunks, faiss_index, bm25_index, INDEX_DIR)

    print(f"\n[build] Done. Total chunks indexed: {len(chunks)}")
