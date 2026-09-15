r"""Embed knowledge base chunks and build a FAISS index.

Run from the project root:
    python rag\index.py
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/chunks.jsonl")
INDEX_PATH = Path("data/faiss.index")
META_PATH = Path("data/chunk_meta.json")

MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks(path: Path) -> list[dict]:
    """Read chunks.jsonl back into a list, preserving file order."""
    if not path.exists():
        raise SystemExit(f"{path} not found. Run rag/ingest.py first.")

    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    """Encode chunk texts into unit-length vectors.

    Returns a float32 array of shape (n_chunks, 384).
    """
    texts = [chunk["text"] for chunk in chunks]

    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        # Scale every vector to length 1 so inner product == cosine similarity.
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # FAISS requires float32. sentence-transformers usually returns it,
    # but casting explicitly means this never silently breaks.
    return vectors.astype("float32")


def build_index(vectors: np.ndarray) -> faiss.Index:
    """Build an exact inner-product index over unit-length vectors."""
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    return index


def save_index(index: faiss.Index, chunks: list[dict],
               index_path: Path, meta_path: Path) -> None:
    """Persist the index and the chunk metadata that maps back to it.

    FAISS stores vectors only. A search returns integer row positions,
    so we must save the chunks in the SAME ORDER they were added —
    row i in the index corresponds to chunks[i] here.
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading model {MODEL_NAME} (first run downloads ~90 MB)...")
    model = SentenceTransformer(MODEL_NAME)

    vectors = embed_chunks(chunks, model)
    print(f"Embedded to shape {vectors.shape}")

    # Sanity check: every vector should have length 1.0 after normalisation.
    norms = np.linalg.norm(vectors, axis=1)
    print(f"Vector norms: min={norms.min():.4f} max={norms.max():.4f}")

    index = build_index(vectors)
    print(f"Index contains {index.ntotal} vectors")

    save_index(index, chunks, INDEX_PATH, META_PATH)
    print(f"Wrote {INDEX_PATH} and {META_PATH}")