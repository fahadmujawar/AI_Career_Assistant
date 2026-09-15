r"""Retrieve the most similar knowledge base chunks for a query.

Run from the project root as a module (required — this file imports from rag.index):
    python -m rag.retrieve "your question here"
    python -m rag.retrieve "your question here" --k 8 --doc-type jds
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.index import INDEX_PATH, META_PATH, MODEL_NAME

DEFAULT_K = 5

# Module-level cache so repeated calls in one process don't reload
# the index or re-instantiate the model (which takes a few seconds).
_model = None
_index = None
_meta = None


def load_resources() -> tuple[SentenceTransformer, faiss.Index, list[dict]]:
    """Load model, index and metadata once, then reuse."""
    global _model, _index, _meta

    if _model is None:
        if not INDEX_PATH.exists() or not META_PATH.exists():
            raise SystemExit(
                f"Missing {INDEX_PATH} or {META_PATH}. Run rag/index.py first."
            )

        _model = SentenceTransformer(MODEL_NAME)
        _index = faiss.read_index(str(INDEX_PATH))

        with META_PATH.open("r", encoding="utf-8") as f:
            _meta = json.load(f)

        # The index and metadata are a matched pair written together.
        # If these disagree, one of them is stale — fail loudly rather
        # than return the wrong text for the right row.
        if _index.ntotal != len(_meta):
            raise SystemExit(
                f"Index has {_index.ntotal} vectors but metadata has "
                f"{len(_meta)} chunks. Rerun rag/index.py."
            )

    return _model, _index, _meta


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """Embed one query as a (1, 384) float32 array of unit length."""
    vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector.astype("float32")


def retrieve(query: str, k: int = DEFAULT_K,
             doc_type: str | None = None) -> list[dict]:
    """Return the k most similar chunks, each with its similarity score.

    doc_type optionally restricts results to one folder ('cvs', 'jds',
    'reference'). Because IndexFlatIP cannot filter, we over-fetch and
    discard, so a narrow filter may return fewer than k results.
    """
    model, index, meta = load_resources()

    query_vector = embed_query(query, model)

    # Over-fetch when filtering; cap at the corpus size.
    fetch = min(k * 5, index.ntotal) if doc_type else min(k, index.ntotal)

    # search() returns two arrays, each shaped (n_queries, fetch):
    # scores (cosine similarity) and row positions in the index.
    scores, positions = index.search(query_vector, fetch)

    results = []
    for score, position in zip(scores[0], positions[0]):
        # FAISS returns -1 to pad when fewer results exist than requested.
        if position == -1:
            continue

        chunk = meta[int(position)]

        if doc_type and chunk["doc_type"] != doc_type:
            continue

        results.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "doc_type": chunk["doc_type"],
            "text": chunk["text"],
            "score": float(score),
        })

        if len(results) == k:
            break

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the knowledge base.")
    parser.add_argument("query", help="the question to search for")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help=f"number of results (default {DEFAULT_K})")
    parser.add_argument("--doc-type", default=None,
                        choices=["cvs", "jds", "reference"],
                        help="restrict results to one document type")
    args = parser.parse_args()

    results = retrieve(args.query, k=args.k, doc_type=args.doc_type)

    if not results:
        print("No results.")
        return

    print(f"\nQuery: {args.query}\n")
    for rank, result in enumerate(results, start=1):
        print(f"[{rank}] score={result['score']:.4f}  {result['chunk_id']}")
        preview = result["text"][:300]
        print(f"    {preview}...\n")


if __name__ == "__main__":
    main()