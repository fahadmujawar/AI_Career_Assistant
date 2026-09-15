from fastapi import FastAPI
from pydantic import BaseModel
from rag.retrieve import retrieve

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    k: int = 5
    doc_type: str | None = None


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