r"""Load knowledge base documents, split into overlapping chunks, save as JSONL.

Run from the project root:
    python rag\ingest.py
"""

import json
from pathlib import Path

KB_DIR = Path("knowledge_base")
OUTPUT_PATH = Path("data/chunks.jsonl")

CHUNK_SIZE = 150    # words per chunk
CHUNK_OVERLAP = 30  # words repeated between adjacent chunks
MIN_CHUNK_WORDS = 20

# Bullet glyphs and similar artefacts left behind by PDF text extraction.
NOISE_CHARS = ["\u2022", "\u25aa", "\u25cf", "\u00b7", "\u2023", "\uf0b7"]


def normalize_text(text: str) -> str:
    """Clean extraction artefacts and collapse all whitespace to single spaces."""
    for ch in NOISE_CHARS:
        text = text.replace(ch, " ")
    # text.split() with no argument splits on ANY run of whitespace,
    # including newlines and tabs, and discards empty strings.
    return " ".join(text.split())


def load_documents(kb_dir: Path) -> list[dict]:
    """Read every .txt file under kb_dir, recursively."""
    documents = []

    for path in sorted(kb_dir.rglob("*.txt")):
        if not path.is_file():
            continue

        # Path relative to the knowledge base, with forward slashes,
        # so chunk IDs look the same on any operating system.
        relative = path.relative_to(kb_dir).as_posix()

        # "jds/mozn_ai_engineer.txt" -> doc_type "jds"
        parts = relative.split("/")
        doc_type = parts[0] if len(parts) > 1 else "root"

        raw = path.read_text(encoding="utf-8")

        documents.append({
            "source": relative,
            "doc_type": doc_type,
            "text": normalize_text(raw),
        })

    return documents


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word windows."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    step = chunk_size - overlap
    chunks = []
    start = 0

    while start < len(words):
        window = words[start:start + chunk_size]

        if len(window) >= MIN_CHUNK_WORDS:
            chunks.append(" ".join(window))

        # If this window already reached the end, stop —
        # otherwise the last chunk would be emitted repeatedly.
        if start + chunk_size >= len(words):
            break

        start += step

    return chunks


def build_chunks(documents: list[dict]) -> list[dict]:
    """Chunk every document and attach metadata to each chunk."""
    all_chunks = []

    for doc in documents:
        pieces = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)

        if not pieces:
            print(f"  WARNING: no chunks produced for {doc['source']} "
                  f"({len(doc['text'].split())} words)")

        for index, piece in enumerate(pieces):
            all_chunks.append({
                "chunk_id": f"{doc['source']}::{index}",
                "source": doc["source"],
                "doc_type": doc["doc_type"],
                "chunk_index": index,
                "text": piece,
            })

    return all_chunks


def save_chunks(chunks: list[dict], path: Path) -> None:
    """Write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            # ensure_ascii=False keeps non-ASCII characters readable
            # in the file instead of writing them as \uXXXX escapes.
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    docs = load_documents(KB_DIR)
    if not docs:
        raise SystemExit(f"No .txt files found under {KB_DIR.resolve()}")

    chunks = build_chunks(docs)
    save_chunks(chunks, OUTPUT_PATH)

    print(f"\n{len(docs)} documents -> {len(chunks)} chunks")

    # Per-type breakdown, so you can see the corpus balance.
    counts = {}
    for chunk in chunks:
        counts[chunk["doc_type"]] = counts.get(chunk["doc_type"], 0) + 1
    for doc_type in sorted(counts):
        print(f"  {doc_type}: {counts[doc_type]} chunks")