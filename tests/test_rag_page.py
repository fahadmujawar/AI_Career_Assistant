"""AppTest checks for the RAG page's with/without RAG comparison.

Needs the full install (the page imports rag.session_corpus), so it is
skipped in the light CI install. No real AI calls and no embedding model:
retrieval results are put straight into session state and call_llm is faked.
"""

import pytest

pytest.importorskip("faiss")
pytest.importorskip("sentence_transformers")

from streamlit.testing.v1 import AppTest

from utils import ai_analysis, ai_client

EMPTY_MESSAGE = "No retrieved chunks come from a different CV."

CV_A = "JANE DOE\njane@example.com\nSKILLS\nPython and SQL\nEXPERIENCE\nBuilt churn models"
CV_B = "JANE DOE\njane@example.com\nPROJECTS\nDeployed a FastAPI service"

OWN_CHUNK = {"source": "a.pdf", "text": "OWN CHUNK from the analysed CV", "score": 0.9}
OTHER_CHUNK = {"source": "b.pdf", "text": "OTHER CHUNK from the second CV", "score": 0.8}


@pytest.fixture
def llm_calls(monkeypatch):
    """Fake call_llm: record every prompt instead of calling Gemini or Groq."""
    calls = []

    def fake_call_llm(provider, prompt, json_mode=False):
        calls.append(prompt)
        return "fake answer", {"provider_used": provider, "fell_back": False, "error": None}

    def fail(*args, **kwargs):
        raise AssertionError("a real AI provider was called")

    monkeypatch.setattr(ai_analysis, "call_llm", fake_call_llm)
    monkeypatch.setattr(ai_analysis, "_call_provider", fail)
    monkeypatch.setattr(ai_client, "is_demo_mode", lambda: False)
    return calls


def run_page(cv_texts, results):
    at = AppTest.from_file("pages/RAG.py", default_timeout=60)
    at.session_state["cv_texts"] = cv_texts
    at.session_state["rag_demo_results"] = results
    at.session_state["rag_demo_query"] = "ML engineer with Python"
    at.session_state["rag_analyze_choice"] = "a.pdf"
    return at.run()


def ask_ai_buttons(at):
    return [b for b in at.button if b.label.startswith("Ask AI")]


def test_prompts_leave_out_chunks_from_the_analysed_cv(llm_calls):
    at = run_page({"a.pdf": CV_A, "b.pdf": CV_B}, [OWN_CHUNK, OTHER_CHUNK])
    assert not at.exception

    without_rag, with_rag = at.code[0].value, at.code[1].value
    assert OTHER_CHUNK["text"] in with_rag
    assert OWN_CHUNK["text"] not in with_rag
    assert OTHER_CHUNK["text"] not in without_rag


def test_step4_prompts_differ_only_by_retrieved_chunks(llm_calls):
    at = run_page({"a.pdf": CV_A, "b.pdf": CV_B}, [OWN_CHUNK, OTHER_CHUNK])
    ask_ai_buttons(at)[0].click().run()
    assert not at.exception

    assert len(llm_calls) == 2
    without_rag, with_rag = llm_calls
    # Both contain the analysed CV; the RAG prompt is the same text plus chunks.
    assert "Built churn models" in without_rag
    assert with_rag.startswith(without_rag)
    added = with_rag[len(without_rag):]
    assert OTHER_CHUNK["text"] in added
    assert OWN_CHUNK["text"] not in with_rag
    # Personal details in the HEADER are not sent.
    assert "jane@example.com" not in with_rag


def test_no_chunks_from_other_cvs_shows_message(llm_calls):
    at = run_page({"a.pdf": CV_A}, [OWN_CHUNK])
    assert not at.exception

    assert any(EMPTY_MESSAGE in info.value for info in at.info)
    assert not ask_ai_buttons(at)
    assert len(at.code) == 0
    assert llm_calls == []
