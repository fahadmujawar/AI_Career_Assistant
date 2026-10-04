# AI Career Assistant

🔗 **[Live Demo](https://ai-career-assistant-fahad.streamlit.app)**

A Streamlit app that analyzes how well a CV matches a job description, using both traditional keyword matching and RAG-grounded, LLM-powered (Gemini/Groq) qualitative analysis and tailoring suggestions.

## Why I built this

I wanted hands-on, genuine experience working with AI tools and APIs, rather than just reading about them. This project let me build a real, working pipeline end to end: parsing PDFs, handling messy real-world text formatting, and integrating an LLM API to do something actually useful with the results.

## What it does

1. **Upload one or more CVs (PDF)** — extracts raw text using PyMuPDF, and parses it into structured sections (summary, skills, experience, education, etc.) using an alias-based header-matching system that handles varied real-world CV formatting.
2. **Paste a job description** — the target role to compare against.
3. **Keyword-based ATS matching** — extracts frequent, meaningful keywords/phrases from the JD (with stopword filtering and unigram/bigram extraction) and checks which appear in the CV, producing a match score.
4. **AI-powered analysis** — sends the parsed CV and JD (plus retrieved grounding context, see below) to an LLM and returns a match score, strengths, gaps, and specific tailoring suggestions.
5. **AI-powered CV tailoring** — suggests reworded versions of existing CV bullet points to better match the job description's language and priorities, without inventing new experience or achievements.
6. **RAG-grounded suggestions** — analysis and tailoring pull relevant passages from a knowledge base (ATS/hiring guidance) into the prompt via a FAISS similarity search. If you've uploaded more than one CV, it also retrieves from your *other* CVs and cites the source filename inline when a suggestion is informed by them (e.g. "from resume_v2.pdf").
7. **Interactive RAG demo page** — a dedicated "RAG" page (Streamlit sidebar) that visualizes the retrieval pipeline step by step over your own uploaded documents: chunk → embed → index → retrieve → see the retrieved text get injected into the real prompt → optional side-by-side generation with vs. without RAG context.
8. **Selectable AI provider with automatic fallback** — choose Gemini or Groq from a sidebar dropdown; if the selected model is rate-limited or temporarily unavailable, the app automatically retries on the other one and tells you when it did.

## Tech stack

- Python, Streamlit
- PyMuPDF (PDF text extraction)
- Google Gemini API (`google-genai`) and Groq (`openai` SDK pointed at Groq's OpenAI-compatible endpoint) — selectable in-app, with automatic fallback between them
- `sentence-transformers` (all-MiniLM-L6-v2) + FAISS for chunk embedding and retrieval (RAG)
- FastAPI + uvicorn — experimental REST API exposing the same RAG/analysis logic outside the Streamlit UI
- Standard library (`re`, `collections.Counter`) for keyword extraction — no heavy NLP dependencies

## Running it locally

1. Clone the repo and set up a virtual environment:
```
python -m venv venv
venv\Scripts\Activate.ps1 # Windows PowerShell
pip install -r requirements.txt
```

2. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com) (no credit card required).
   Optionally, also get a free Groq API key from [console.groq.com/keys](https://console.groq.com/keys) — the app can use either model, and automatically falls back to the other one if your selected model is rate-limited or temporarily unavailable.
3. Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```
(Groq is optional — the app works with just a Gemini key, and vice versa. Deployed on Streamlit Cloud, add the same keys under the app's Secrets instead of a `.env` file.)

4. Build the search index (run these from the project root, the folder that contains `app.py`):
```
python -m rag.ingest
python -m rag.index
```
Without this step the app still runs, but quietly without RAG grounding. The first run downloads the embedding model (about 90 MB). Rerun both commands after adding or editing files in `knowledge_base/`.

5. Run the app:
`streamlit run app.py`

## API backend (experimental)

`api/main.py` exposes the same RAG retrieval and CV/JD analysis logic as a REST API, independent of the Streamlit UI:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check |
| `POST /rag/query` | Search the static `knowledge_base/` FAISS index (`query`, `k`, `doc_type`) |
| `POST /rag/session-query` | Build a one-off in-memory index from a submitted `cv_text` + `jd_text` and search it (`query`, `k`) |
| `POST /analyse` | Parse `cv_text`, analyze it against `jd_text` with the chosen `provider` (Gemini/Groq), and return the same match score/strengths/gaps/suggestions as the UI |

Run it locally with:
```
venv\Scripts\uvicorn.exe api.main:app --reload
```
`fastapi` is included in `requirements.txt`.

## Testing and Evaluation

### Running the tests
```
pip install pytest
python -m pytest
```
The tests need no API keys.

They cover the keyword matcher, document chunking, the CV parser, the check on the AI's JSON reply, switching to the other model when one fails, and the RAG page's with and without RAG prompts. Locally: 30 passed and 3 expected failures. Three known bugs are kept as xfail tests:
- a phone number keeps the line break after it
- a heading with a colon, such as "SKILLS:", is not detected
- keyword pairs join across the end of a sentence

GitHub Actions runs the tests on every push and pull request with Python 3.11 (`.github/workflows/tests.yml`). It installs only what the tests need, so the RAG page tests are skipped there.

### Retrieval
On 12 labelled questions, the expected reference file was the first result 12/12 times (hit@1) and in the top 4 results 12/12 times (hit@4). There are only 4 reference files, on clearly different topics, so this shows the pipeline works, not that it scales.

### Match score consistency
10 runs on the same CV and JD were planned, 5 per model. Gemini returned 503 on every call, so Groq answered all 10. 9 gave a valid score (70 to 83, average about 77.7); 1 reply failed the Pydantic check. There is no Gemini comparison yet.

### What the real run showed
- The fallback switched models correctly.
- The Pydantic check caught a Groq reply missing two fields.
- Tailoring on Groq failed with a 400 JSON error.

Full results are in `eval/results.md`.

### Evaluation scripts
- `python eval/eval_retrieval.py` runs the retrieval questions in `eval/retrieval_cases.json` (needs the search index built, see "Running it locally").
- `python eval/eval_scores.py --dry-run` shows the plan without calling any API. A real run, without `--dry-run`, uses about 12 to 20 API calls.

### Next steps
- Test retrieval on more documents, including ones with overlapping topics.
- Report MRR (mean reciprocal rank) alongside hit@1 and hit@4.
- Average several runs for each match score.

## Known limitations

- The keyword matcher matches whole words, but not word variants such as plurals ("client" vs "clients"). This is intentionally deferred, since the AI analysis handles this kind of matching.
- The API has no authentication.
- AI match scores vary between runs: 70 to 83 on the same input.
- Uploaded CVs are kept only in memory and are never saved.
- Section-header detection relies on an alias list built from CV formats I've personally tested against — it may not recognize header wording it hasn't seen before.

## What I'd build next

- Refactor from dictionaries/`session_state` to proper Python classes (`Resume`, `JobDescription`, `ATSReport`) as the codebase grows
- Expand the section-header alias list based on more real-world CV formats