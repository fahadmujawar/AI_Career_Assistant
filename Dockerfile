FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    fastapi==0.141.1 \
    uvicorn==0.52.0 \
    faiss-cpu \
    sentence-transformers \
    google-genai \
    openai \
    python-dotenv \
    pydantic \
    python-multipart

# Bake the embedding model into the image so containers don't need
# network access to Hugging Face Hub on their first request.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY api/ ./api/
COPY rag/ ./rag/
COPY utils/ ./utils/
COPY knowledge_base/reference/ ./knowledge_base/reference/
RUN python -m rag.ingest && python -m rag.index

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]