r"""Build an ephemeral, in-memory retrieval index from documents uploaded
during the current Streamlit session (CVs from pages/RAG.py or the main
app), instead of the static files under knowledge_base/.

Reuses the same chunking (rag.ingest) and indexing (rag.index) building
blocks as the file-based pipeline -- only the source of the text and
where the index lives (in st.session_state, not on disk) differ.
"""

import streamlit as st

from rag.index import MODEL_NAME, build_index
from rag.ingest import CHUNK_OVERLAP, CHUNK_SIZE, chunk_text, normalize_text


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Load the sentence-transformers model once per app process."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def build_chunks_from_texts(sources: dict, doc_type: str = "cv") -> list[dict]:
    """Chunk each {name: raw_text} entry into the same dict shape rag/ingest.py
    produces, so downstream code (prompt building, display) doesn't care
    whether a chunk came from disk or from an upload.
    """
    chunks = []

    for name, raw_text in sources.items():
        text = normalize_text(raw_text)
        pieces = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for index, piece in enumerate(pieces):
            chunks.append({
                "chunk_id": f"{name}::{index}",
                "source": name,
                "doc_type": doc_type,
                "chunk_index": index,
                "text": piece,
            })

    return chunks


def build_session_index(chunks: list[dict]):
    """Embed chunks and build an in-memory FAISS index. Returns (index, chunks).

    Returns (None, []) for an empty corpus rather than erroring -- callers
    should treat "nothing to index yet" as a normal state, not a failure.
    """
    if not chunks:
        return None, []

    model = get_embedding_model()
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    return build_index(vectors), chunks


def search_session_index(query: str, index, chunks: list[dict], k: int = 5) -> list[dict]:
    """Return the k chunks most similar to query, each with its score."""
    if index is None or not chunks:
        return []

    model = get_embedding_model()
    query_vector = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    fetch = min(k, index.ntotal)
    scores, positions = index.search(query_vector, fetch)

    results = []
    for score, position in zip(scores[0], positions[0]):
        if position == -1:
            continue
        chunk = chunks[int(position)]
        results.append({**chunk, "score": float(score)})

    return results
