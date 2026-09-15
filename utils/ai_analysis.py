import json

from google.genai import errors as genai_errors
from openai import OpenAIError

from utils.ai_client import GEMINI_MODEL, GROQ_MODEL, get_gemini_client, get_groq_client

try:
    from rag.retrieve import retrieve, INDEX_PATH, META_PATH
except ImportError:
    # RAG is optional grounding, not a hard dependency of the app --
    # faiss/sentence-transformers may not be installed everywhere.
    retrieve = None
    INDEX_PATH = META_PATH = None


def get_relevant_context(job_description, k=4):
    """Fetch grounding chunks from the reference knowledge base for this JD.

    Returns "" if RAG dependencies aren't installed, or the FAISS index
    hasn't been built yet (data/faiss.index is a local build artifact,
    produced by rag/ingest.py + rag/index.py, not committed to git).
    Callers must treat this as optional -- the app has to keep working
    without it.
    """
    if retrieve is None or not INDEX_PATH.exists() or not META_PATH.exists():
        return ""

    try:
        results = retrieve(job_description, k=k, doc_type="reference")
    except Exception:
        return ""

    if not results:
        return ""

    return "\n\n".join(f"[{r['source']}]\n{r['text']}" for r in results)


def get_cross_cv_context(job_description, cv_texts, exclude_name=None, k=4):
    """Retrieve chunks relevant to job_description from the user's OTHER
    uploaded CVs (every key in cv_texts except exclude_name), so a
    suggestion can cite exactly which CV it was informed by.

    Returns "" if there's nothing to search -- one CV or none uploaded,
    or the session-corpus/RAG dependencies aren't available. This mirrors
    get_relevant_context()'s "optional, never breaks the app" contract.
    """
    other_texts = {
        name: text for name, text in (cv_texts or {}).items() if name != exclude_name
    }
    if not other_texts:
        return ""

    try:
        from rag.session_corpus import (
            build_chunks_from_texts,
            build_session_index,
            search_session_index,
        )
    except ImportError:
        return ""

    try:
        chunks = build_chunks_from_texts(other_texts, doc_type="cv")
        index, chunks = build_session_index(chunks)
        results = search_session_index(job_description, index, chunks, k=k)
    except Exception:
        return ""

    if not results:
        return ""

    return "\n\n".join(f"[{r['source']}]\n{r['text']}" for r in results)


def build_analysis_prompt(
    resume_sections, job_description, reference_context="", other_cv_context=""
):
    relevant_sections = {
        key: "\n".join(lines)
        for key, lines in resume_sections.items()
        if key != "HEADER"
    }

    resume_text = "\n\n".join(
        f"{section}:\n{content}"
        for section, content in relevant_sections.items()
    )

    context_block = ""
    if reference_context:
        context_block += f"""
REFERENCE MATERIAL (hiring/ATS guidance -- use to inform your judgment,
do not quote it directly or treat it as part of the CV or JD):
{reference_context}
"""
    if other_cv_context:
        context_block += f"""
OTHER CVs THIS CANDIDATE HAS UPLOADED (each chunk below is labeled with
its source filename in square brackets). If a strength, gap, or
suggestion is informed by content found here, cite it inline using the
exact filename, e.g. "(from resume_v2.pdf)" -- do not cite a filename
you don't see below:
{other_cv_context}
"""

    prompt = f"""
You are an assistant helping a job seeker understand how well their CV matches a job description.

CV CONTENT:
{resume_text}

JOB DESCRIPTION:
{job_description}
{context_block}
Analyze the fit between the CV and the job description. Respond with ONLY a valid JSON object,
no extra text, no markdown code fences, in exactly this shape:

{{
  "match_score": <integer 0-100>,
  "strengths": ["<specific strength 1>", "<specific strength 2>", ...],
  "gaps": ["<specific gap or missing requirement 1>", ...],
  "suggestions": ["<specific, actionable suggestion 1>", ...]
}}

Keep each list to 3-6 items. Be specific and reference actual content from the CV and JD,
not generic advice.
"""
    return prompt


def _call_provider(provider, prompt, json_mode=False):
    """Call one provider with a plain-text prompt.

    Returns (text, error): text is None on failure and error is a short,
    user-facing reason. Never raises -- callers (call_llm) decide whether
    to show the failure or fall back to another provider.
    """
    if provider == "Gemini":
        try:
            client = get_gemini_client()
        except ValueError as e:
            return None, str(e)

        try:
            config = {"response_mime_type": "application/json"} if json_mode else {}
            response = client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt, config=config
            )
            return response.text, None
        except genai_errors.APIError as e:
            return None, f"Gemini is unavailable ({e.code} {e.status})."
        except Exception as e:
            return None, f"Gemini request failed: {e}"

    if provider == "Groq":
        try:
            client = get_groq_client()
        except ValueError as e:
            return None, str(e)

        try:
            kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return response.choices[0].message.content, None
        except OpenAIError as e:
            return None, f"Groq is unavailable ({e})."
        except Exception as e:
            return None, f"Groq request failed: {e}"

    return None, f"Unknown AI provider: {provider}"


def call_llm(provider, prompt, json_mode=False):
    """Call `provider`; if it fails, automatically retry on the other
    supported provider so one model's outage doesn't break the app.

    Returns (text, meta). text is None only if BOTH providers failed
    (meta["error"] explains why). meta always has "provider_used" and
    "fell_back", so callers can tell the user when a fallback happened.
    """
    other = "Groq" if provider == "Gemini" else "Gemini"

    text, error = _call_provider(provider, prompt, json_mode=json_mode)
    if text is not None:
        return text, {"provider_used": provider, "fell_back": False, "error": None}

    fallback_text, fallback_error = _call_provider(other, prompt, json_mode=json_mode)
    if fallback_text is not None:
        return fallback_text, {"provider_used": other, "fell_back": True, "error": None}

    return None, {
        "provider_used": provider,
        "fell_back": False,
        "error": f"{provider} failed ({error}). Fallback {other} also failed ({fallback_error}).",
    }


def _call_llm_json(provider, prompt):
    """Call an LLM expecting a JSON response; always returns a dict.

    On failure (both providers down, or a non-JSON reply), returns an
    {"error": ...} dict. On success, adds "_provider_requested",
    "_provider_used" and "_fell_back" so callers can tell the user when
    a fallback happened -- pop these before treating the dict as the
    plain analysis/tailoring result.
    """
    text, meta = call_llm(provider, prompt, json_mode=True)

    if text is None:
        return {"error": meta["error"], "raw_response": ""}

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Could not parse AI response as JSON.", "raw_response": text}

    result["_provider_requested"] = provider
    result["_provider_used"] = meta["provider_used"]
    result["_fell_back"] = meta["fell_back"]
    return result


def analyze_cv_against_jd(
    resume_sections, job_description, provider="Gemini", cv_texts=None, current_cv_name=None
):
    reference_context = get_relevant_context(job_description)
    other_cv_context = get_cross_cv_context(job_description, cv_texts, exclude_name=current_cv_name)
    prompt = build_analysis_prompt(
        resume_sections, job_description, reference_context, other_cv_context
    )

    return _call_llm_json(provider, prompt)


def build_tailoring_prompt(
    resume_sections, job_description, reference_context="", other_cv_context=""
):
    relevant_sections = {
        key: "\n".join(lines)
        for key, lines in resume_sections.items()
        if key != "HEADER"
    }

    resume_text = "\n\n".join(
        f"{section}:\n{content}"
        for section, content in relevant_sections.items()
    )

    context_block = ""
    if reference_context:
        context_block += f"""
REFERENCE MATERIAL (hiring/ATS guidance -- use to inform wording and
terminology choices, do not quote it directly or treat it as part of
the CV or JD):
{reference_context}
"""
    if other_cv_context:
        context_block += f"""
OTHER CVs THIS CANDIDATE HAS UPLOADED (each chunk below is labeled with
its source filename in square brackets). If a rewrite borrows wording,
phrasing, or an achievement from one of these, cite it inline in the
rewritten text using the exact filename, e.g. "(from resume_v2.pdf)" --
do not cite a filename you don't see below:
{other_cv_context}
"""

    prompt = f"""
You are helping a job seeker tailor their existing CV bullet points to better match a job description,
WITHOUT inventing new experience, skills, or achievements they did not actually have.

CV CONTENT:
{resume_text}

JOB DESCRIPTION:
{job_description}
{context_block}
For each section below, suggest rewritten versions of the EXISTING bullet points that:
- Keep the same underlying facts, tools, and achievements (do not invent numbers, tools, or outcomes)
- Reframe emphasis and wording to better match the job description's priorities and terminology
- Stay truthful to what is actually written in the original CV content

Only include sections that have bullet points worth revising (skip EDUCATION and CERTIFICATIONS
unless there's a genuinely useful rewording).

Respond with ONLY a valid JSON object, no extra text, no markdown code fences, in exactly this shape:

{{
  "SECTION_NAME": [
    {{"original": "<original bullet text>", "rewritten": "<tailored rewrite>"}},
    ...
  ],
  ...
}}
"""
    return prompt


def tailor_cv_to_jd(
    resume_sections, job_description, provider="Gemini", cv_texts=None, current_cv_name=None
):
    reference_context = get_relevant_context(job_description)
    other_cv_context = get_cross_cv_context(job_description, cv_texts, exclude_name=current_cv_name)
    prompt = build_tailoring_prompt(
        resume_sections, job_description, reference_context, other_cv_context
    )

    return _call_llm_json(provider, prompt)