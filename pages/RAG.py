r"""RAG — a hands-on, step-by-step demo of the retrieval pipeline, run over
YOUR uploaded CVs and job description -- not a fixed reference corpus.

This is a page in the same app as app.py (Streamlit auto-discovers
pages/*.py). CVs uploaded on the main page are already visible here via
st.session_state; this page also lets you add more directly.
"""

import streamlit as st

from rag.session_corpus import (
    build_chunks_from_texts,
    build_session_index,
    search_session_index,
)
from utils.ai_analysis import build_analysis_prompt, call_llm
from utils.ai_client import is_demo_mode, render_provider_selector
from utils.parser import extract_pdf_text
from utils.resume_parser import parse_resume

st.set_page_config(page_title="RAG", page_icon="🔎", layout="wide")

st.title("RAG")
st.caption(
    "Retrieval-Augmented Generation, made visible — run over your own "
    "uploaded CVs and job description, the same way it grounds this app's "
    "AI analysis and tailoring features."
)

selected_provider = render_provider_selector()

with st.expander("What is RAG, in one paragraph?", expanded=True):
    st.markdown(
        """
Instead of asking an LLM to answer purely from what it memorized during
training, **RAG retrieves relevant passages from your own documents first**,
then hands them to the model as extra context before it generates an
answer. Three steps, each shown below: **chunk + embed** your documents
into a searchable index → **retrieve** the closest chunks to a query →
**inject** them into the LLM prompt before it generates a response.
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Section 1 — your documents (the corpus)
# ---------------------------------------------------------------------------
st.header("1. Your documents — this is the corpus")

st.caption(
    "Retrieval here runs over the CVs you've uploaded, not a fixed "
    "reference corpus — upload one on the main page's 'Upload Master CVs' "
    "step, or add one below."
)

cv_texts = st.session_state.setdefault("cv_texts", {})

if cv_texts:
    st.write(f"**{len(cv_texts)} CV(s) available:**")
    for name, text in cv_texts.items():
        st.write(f"- `{name}` — {len(text):,} characters")
else:
    st.info("No CVs uploaded yet. Add at least one below to get started.")

new_files = st.file_uploader(
    "Add more CVs",
    type=["pdf"],
    accept_multiple_files=True,
    key="rag_page_cv_uploader",
)

if new_files:
    added = [f for f in new_files if f.name not in cv_texts]
    for f in added:
        text, _ = extract_pdf_text(f)
        cv_texts[f.name] = text
    if added:
        st.success(f"Added {len(added)} new CV(s).")
        st.rerun()

job_description = st.text_area(
    "Job description (used as the analysis prompt's JD, and as the "
    "default retrieval query below)",
    height=150,
    key="job_description",
)

st.divider()

st.subheader("Session index — chunk → embed → FAISS (in memory only)")
st.caption(
    "Rebuild this whenever you add or change a CV. It lives only in this "
    "browser session, unlike knowledge_base/'s index which is a file on disk."
)

if st.button(
    "Build / refresh the session index", type="primary", disabled=not cv_texts
):
    with st.spinner("Chunking and embedding your CVs..."):
        chunks = build_chunks_from_texts(cv_texts, doc_type="cv")
        index, chunks = build_session_index(chunks)
        st.session_state.rag_session_index = index
        st.session_state.rag_session_chunks = chunks
        # A stale retrieval from before the rebuild would be misleading.
        st.session_state.pop("rag_demo_results", None)
    st.success(f"Indexed {len(chunks)} chunks from {len(cv_texts)} CV(s).")

index = st.session_state.get("rag_session_index")
indexed_chunks = st.session_state.get("rag_session_chunks", [])

st.divider()

# ---------------------------------------------------------------------------
# Section 2 — retrieve
# ---------------------------------------------------------------------------
st.header("2. Retrieve — find the closest chunks across your CVs")

if index is None:
    st.info("Build the session index above first.")
else:
    query = st.text_area(
        "Query (defaults to the job description above — edit freely)",
        value=st.session_state.get("job_description", ""),
        height=100,
        key="rag_query",
    )
    k = st.slider("How many chunks to retrieve (k)", 1, 10, 4)

    if st.button("Retrieve", type="primary") and query.strip():
        with st.spinner("Embedding the query and searching..."):
            st.session_state.rag_demo_results = search_session_index(
                query, index, indexed_chunks, k=k
            )
            st.session_state.rag_demo_query = query

    results = st.session_state.get("rag_demo_results")

    if results:
        st.write(f"**{len(results)} nearest chunks** (cosine similarity):")
        for rank, r in enumerate(results, start=1):
            st.markdown(
                f"**#{rank} — `{r['source']}`**  ·  similarity: **{r['score']:.3f}**"
            )
            st.progress(min(max(r["score"], 0.0), 1.0))
            with st.expander("Show chunk text"):
                st.write(r["text"])
    elif results == []:
        st.warning("No chunks matched. Try a broader query.")

st.divider()

# ---------------------------------------------------------------------------
# Section 3 — see the retrieved chunks change the prompt
# ---------------------------------------------------------------------------
st.header("3. Augment — injecting retrieved context into the LLM prompt")

results = st.session_state.get("rag_demo_results")
demo_query = st.session_state.get("rag_demo_query", "")

if not results:
    st.info("Run a retrieval query in step 2 above to see the prompt difference.")
elif not cv_texts:
    st.info("Upload a CV in step 1 to build the analysis prompt against.")
else:
    analyze_choice = st.selectbox(
        "Which uploaded CV is this analysis for?",
        list(cv_texts.keys()),
        key="rag_analyze_choice",
    )
    resume_sections = parse_resume(cv_texts[analyze_choice])["sections"]
    other_cv_context = "\n\n".join(f"[{r['source']}]\n{r['text']}" for r in results)

    st.caption(
        "Notice the retrieved chunks can come from a *different* uploaded "
        "CV than the one being analyzed — RAG surfaces the best-matching "
        "material across everything you've uploaded, and the prompt asks "
        "the model to cite the source filename when it uses it."
    )

    tab_without, tab_with = st.tabs(
        ["Prompt WITHOUT RAG", "Prompt WITH RAG (what actually gets sent)"]
    )
    with tab_without:
        st.code(
            build_analysis_prompt(resume_sections, demo_query),
            language="text",
        )
    with tab_with:
        st.code(
            build_analysis_prompt(
                resume_sections, demo_query, other_cv_context=other_cv_context
            ),
            language="text",
        )
        st.caption(
            "The OTHER CVs block above is exactly what step 2's retrieval "
            "added — this is the real build_analysis_prompt() function "
            "from utils/ai_analysis.py, not a re-implementation."
        )

st.divider()

# ---------------------------------------------------------------------------
# Section 4 — generate, with vs. without RAG
# ---------------------------------------------------------------------------
st.header("4. Generate — does the retrieved context change the answer?")

if is_demo_mode():
    st.info(
        "Live AI calls are disabled on the public demo to protect shared "
        "free-tier quotas. Clone the repo and add your own API key to try "
        "this step yourself — see the README."
    )
elif not results or not cv_texts:
    st.info("Complete steps 1-3 above first.")
else:
    if st.button("Ask AI — with vs. without RAG context"):
        base_prompt = (
            "In 2-3 sentences, advise how this candidate should position "
            f"themselves for this job description:\n\n{demo_query}"
        )
        rag_prompt = (
            f"{base_prompt}\n\nGround your advice in this content from their "
            f"other uploaded CVs where relevant (each chunk is labeled with "
            f"its source filename in square brackets) -- if you use one, "
            f'cite it inline, e.g. "(from resume_v2.pdf)":\n{other_cv_context}'
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Without RAG")
            with st.spinner("Generating..."):
                text, meta = call_llm(selected_provider, base_prompt)
            if text is None:
                st.error(meta["error"])
            else:
                if meta["fell_back"]:
                    st.warning(
                        f"⚠️ {selected_provider} was unavailable — used "
                        f"{meta['provider_used']} instead."
                    )
                st.write(text)

        with col_b:
            st.subheader("With RAG")
            with st.spinner("Generating..."):
                text, meta = call_llm(selected_provider, rag_prompt)
            if text is None:
                st.error(meta["error"])
            else:
                if meta["fell_back"]:
                    st.warning(
                        f"⚠️ {selected_provider} was unavailable — used "
                        f"{meta['provider_used']} instead."
                    )
                st.write(text)

        st.caption(
            "Same question, same model — the only difference is whether the "
            "retrieved reference material was in the prompt."
        )
