# RAG Module

Retrieval-Augmented Generation for the UQ BIT evaluation pipeline.

The module builds a hybrid search index over UQ source documents and exposes a `Retriever` class that the evaluation scripts use to inject relevant passages into model prompts.

## Prerequisites

Install all dependencies from the project root:

```powershell
pip install -r requirements.txt
```

The RAG-specific packages are:

| Package | Purpose |
|---------|---------|
| `sentence-transformers` | BGE-M3 embedder + bge-reranker-v2-m3 cross-encoder |
| `faiss-cpu` | Dense vector index (FAISS IndexFlatIP) |
| `rank_bm25` | Sparse BM25 retrieval |
| `pypdf` | PDF text extraction |
| `beautifulsoup4` | HTML cleaning for URL ingest |

An `OPENAI_API_KEY` is required for the Contextual Retrieval step (GPT-4o-mini writes a 1-sentence context prefix per chunk). Skip it with `--no-contextualise` if you want a no-cost build at the cost of lower retrieval quality.

## Corpus

The index is built from two sources:

**PDFs** - all `*.pdf` files in `data-collection/sources/`:
- `international-guide-undergraduate-postgraduate.pdf` - UQ International Student Guide 2026
- `domestic-guide-undergraduate.pdf` - UQ Domestic Undergraduate Guide 2026

Adding a new PDF is automatic: drop it into `data-collection/sources/` and re-run `python -m rag build --force`.

**Program pages** - all URLs listed in `data-collection/sources/program-pages.txt` (8 UQ BIT program pages).

## Building the index

Run from the **project root** (not from inside `rag/`):

```powershell
# Full build - PDF + URLs + contextualisation + embed + persist
python -m rag build

# Rebuild from scratch (overwrites existing index)
python -m rag build --force

# Skip GPT-4o-mini contextualisation (no API cost, lower retrieval quality)
python -m rag build --no-contextualise
```

Output goes to `rag/index/` (gitignored):

| File | Contents |
|------|---------|
| `chunks.jsonl` | One JSON record per chunk: `{id, source, page, text, context_prefix}` |
| `dense.faiss` | FAISS IndexFlatIP over BGE-M3 embeddings |
| `bm25.pkl` | Serialised BM25Okapi index |
| `context_cache.json` | GPT-4o-mini prefix cache (keyed by chunk-text hash) |

Expected corpus size: 500-1500 chunks. Build time: ~5-15 min depending on whether context prefixes are cached.

## Querying the index

```powershell
# Top-5 results (default)
python -m rag query "What is the minimum ATAR for the BIT?"

# Custom top-k
python -m rag query "What majors are available?" --top-k 10
```

Each result shows the source file/URL, page number (for PDFs), and the retrieved chunk text.

## Pipeline details

Retrieval follows four SOTA stages:

1. **Dense search** - query embedded with BGE-M3, FAISS inner-product search returns top-20
   (Chen et al. 2024, arXiv:2402.03216)
2. **Sparse search** - BM25Okapi over tokenised chunks returns top-20
3. **RRF fusion** - Reciprocal Rank Fusion merges the two lists
   (Cormack, Clarke & Buttcher 2009, SIGIR)
4. **Cross-encoder rerank** - `bge-reranker-v2-m3` scores the fused top-20 and returns the final top-k
   (Li et al. 2023, arXiv:2312.15503)

Index build uses Contextual Retrieval: GPT-4o-mini prepends a 1-sentence context summary to each chunk before embedding, reducing retrieval failures by ~49% (Anthropic 2024).

## Integration with evaluate.py

The evaluation scripts instantiate `Retriever` once before the model loop and pass it through to each RAG variant. Use `--no-rag` to skip all RAG variants:

```powershell
# 16-config run with RAG (default — index must be built first)
python fine-tuning/gemma3-12b-grpo/evaluate.py

# 8-config run without RAG
python fine-tuning/gemma3-12b-grpo/evaluate.py --no-rag

# Custom index location
python fine-tuning/gemma3-12b-grpo/evaluate.py --index-dir /path/to/rag/index
```

If the index is missing or dependencies are not installed, RAG variants are skipped automatically with a warning and the non-RAG variants still run.
