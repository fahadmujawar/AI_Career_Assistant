# AI Career Assistant

🔗 **[Live Demo](https://ai-career-assistant-fahad.streamlit.app)**

A Streamlit app that analyzes how well a CV matches a job description, using both traditional keyword matching and Gemini-powered qualitative analysis and tailoring suggestions.

## Why I built this

I wanted hands-on, genuine experience working with AI tools and APIs, rather than just reading about them. This project let me build a real, working pipeline end to end: parsing PDFs, handling messy real-world text formatting, and integrating an LLM API to do something actually useful with the results.

## What it does

1. **Upload one or more CVs (PDF)** — extracts raw text using PyMuPDF, and parses it into structured sections (summary, skills, experience, education, etc.) using an alias-based header-matching system that handles varied real-world CV formatting.
2. **Paste a job description** — the target role to compare against.
3. **Keyword-based ATS matching** — extracts frequent, meaningful keywords/phrases from the JD (with stopword filtering and unigram/bigram extraction) and checks which appear in the CV, producing a match score.
4. **AI-powered analysis (Gemini)** — sends the parsed CV and JD to Google's Gemini API and returns a match score, strengths, gaps, and specific tailoring suggestions.
5. **AI-powered CV tailoring (Gemini)** — suggests reworded versions of existing CV bullet points to better match the job description's language and priorities, without inventing new experience or achievements.

## Tech stack

- Python, Streamlit
- PyMuPDF (PDF text extraction)
- Google Gemini API (`google-genai`) and Groq (`openai` SDK pointed at Groq's OpenAI-compatible endpoint) — selectable in-app, with automatic fallback between them
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

4. Run the app:
`streamlit run app.py`

## Known limitations

- The keyword matcher does exact string matching after basic cleaning — it doesn't handle word variants (e.g. "client" vs "clients"). This is intentionally deferred, since the Gemini-based analysis naturally handles this kind of semantic matching.
- Section-header detection relies on an alias list built from CV formats I've personally tested against — it may not recognize header wording it hasn't seen before.

## What I'd build next

- Refactor from dictionaries/`session_state` to proper Python classes (`Resume`, `JobDescription`, `ATSReport`) as the codebase grows
- Expand the section-header alias list based on more real-world CV formats