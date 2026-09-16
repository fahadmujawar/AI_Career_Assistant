import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import faiss
from rag.index import MODEL_NAME, build_index
from rag.ingest import CHUNK_OVERLAP, CHUNK_SIZE, chunk_text, normalize_text
from rag.retrieve import retrieve
from utils.ai_analysis import analyze_cv_against_jd
from utils.resume_parser import parse_resume


app = FastAPI()

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ---------- request models ----------

class QueryRequest(BaseModel):
    query: str
    k: int = 5
    doc_type: str | None = None


class SessionQueryRequest(BaseModel):
    cv_text: str
    jd_text: str
    query: str
    k: int = 5

class AnalyseRequest(BaseModel):
    cv_text: str
    jd_text: str
    provider: str = "Gemini"


# ---------- endpoints ----------

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/rag/query")
def rag_query(request: QueryRequest):
    results = retrieve(
        query=request.query,
        k=request.k,
        doc_type=request.doc_type
    )
    return {"results": results}


@app.post("/rag/session-query")
def session_query(request: SessionQueryRequest):
    model = get_model()

    # Build chunks from only what the user sent
    sources = {
        "uploaded_cv": request.cv_text,
        "uploaded_jd": request.jd_text,
    }

    chunks = []
    for name, raw_text in sources.items():
        text = normalize_text(raw_text)
        pieces = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, piece in enumerate(pieces):
            chunks.append({
                "chunk_id": f"{name}::{i}",
                "source": name,
                "doc_type": "cv" if "cv" in name else "jd",
                "chunk_index": i,
                "text": piece,
            })

    if not chunks:
        return {"results": []}

    # Embed all chunks into vectors
    texts = [chunk["text"] for chunk in chunks]
    chunk_vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    # Build a brand new in-memory index from scratch
    dimension = chunk_vectors.shape[1]
    session_index = faiss.IndexFlatIP(dimension)
    session_index.add(chunk_vectors)

    # Embed the query
    query_vector = model.encode(
        [request.query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    # Search only this session index
    fetch = min(request.k, session_index.ntotal)
    scores, positions = session_index.search(query_vector, fetch)

    results = []
    for score, position in zip(scores[0], positions[0]):
        if position == -1:
            continue
        chunk = chunks[int(position)]
        results.append({**chunk, "score": float(score)})

    return {"results": results}

@app.post("/analyse")
def analyse(request: AnalyseRequest):
    resume = parse_resume(request.cv_text)
    result = analyze_cv_against_jd(
        resume_sections=resume["sections"],
        job_description=request.jd_text,
        provider=request.provider,
    )
    return result
    