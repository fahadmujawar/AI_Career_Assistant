import json

import pytest

from utils import ai_analysis
from utils.ai_analysis import AnalysisResult, TailoringResult

GOOD_ANALYSIS = {
    "match_score": 85,
    "strengths": ["Python"],
    "gaps": ["Kubernetes"],
    "suggestions": ["Mention Docker"],
}


@pytest.fixture(autouse=True)
def no_real_api_calls(monkeypatch):
    """Fail any test that would reach Gemini or Groq."""
    def fail(*args, **kwargs):
        raise AssertionError("a real AI provider was called")

    monkeypatch.setattr(ai_analysis, "_call_provider", fail)


@pytest.fixture
def fake_reply(monkeypatch):
    """Make call_llm return the given reply as if Groq answered after a fallback."""
    def set_reply(reply):
        text = reply if isinstance(reply, str) else json.dumps(reply)

        def fake_call_llm(provider, prompt, json_mode=False):
            return text, {"provider_used": "Groq", "fell_back": True, "error": None}

        monkeypatch.setattr(ai_analysis, "call_llm", fake_call_llm)
        return text

    return set_reply


def assert_error_format(result, raw_text):
    assert set(result) == {"error", "raw_response"}
    assert result["raw_response"] == raw_text


# ---------- reply validation ----------

def test_good_analysis_passes_and_provider_keys_are_added(fake_reply):
    fake_reply(GOOD_ANALYSIS)
    result = ai_analysis._call_llm_json("Gemini", "prompt", AnalysisResult)

    assert "error" not in result
    assert result["match_score"] == 85
    assert result["_provider_requested"] == "Gemini"
    assert result["_provider_used"] == "Groq"
    assert result["_fell_back"] is True


def test_missing_part_returns_error_format(fake_reply):
    reply = {k: v for k, v in GOOD_ANALYSIS.items() if k != "gaps"}
    text = fake_reply(reply)
    result = ai_analysis._call_llm_json("Gemini", "prompt", AnalysisResult)

    assert_error_format(result, text)
    assert "gaps" in result["error"]


def test_json_list_returns_error_format(fake_reply):
    text = fake_reply([GOOD_ANALYSIS])
    result = ai_analysis._call_llm_json("Gemini", "prompt", AnalysisResult)

    assert_error_format(result, text)


def test_not_json_returns_error_format(fake_reply):
    text = fake_reply("Sorry, I cannot help with that.")
    result = ai_analysis._call_llm_json("Gemini", "prompt", AnalysisResult)

    assert_error_format(result, text)


@pytest.mark.parametrize("score", ["85", 85.0])
def test_score_as_text_or_whole_float_becomes_int(fake_reply, score):
    fake_reply({**GOOD_ANALYSIS, "match_score": score})
    result = ai_analysis._call_llm_json("Gemini", "prompt", AnalysisResult)

    assert result["match_score"] == 85
    assert type(result["match_score"]) is int


@pytest.mark.parametrize("score", [85.5, "high", 101, -1])
def test_bad_score_returns_error_format(fake_reply, score):
    text = fake_reply({**GOOD_ANALYSIS, "match_score": score})
    result = ai_analysis._call_llm_json("Gemini", "prompt", AnalysisResult)

    assert_error_format(result, text)


def test_good_tailoring_passes(fake_reply):
    fake_reply({"EXPERIENCE": [{"original": "Built models", "rewritten": "Built ML models"}]})
    result = ai_analysis._call_llm_json("Gemini", "prompt", TailoringResult)

    assert result["EXPERIENCE"][0]["rewritten"] == "Built ML models"
    assert result["_provider_used"] == "Groq"


def test_tailoring_missing_rewritten_returns_error_format(fake_reply):
    text = fake_reply({"EXPERIENCE": [{"original": "Built models"}]})
    result = ai_analysis._call_llm_json("Gemini", "prompt", TailoringResult)

    assert_error_format(result, text)


# ---------- provider fallback ----------

def test_fallback_to_groq_when_gemini_fails(monkeypatch):
    def fake_provider(provider, prompt, json_mode=False):
        if provider == "Gemini":
            return None, "Gemini is unavailable (503 UNAVAILABLE)."
        return "groq answer", None

    monkeypatch.setattr(ai_analysis, "_call_provider", fake_provider)
    text, meta = ai_analysis.call_llm("Gemini", "prompt")

    assert text == "groq answer"
    assert meta["provider_used"] == "Groq"
    assert meta["fell_back"] is True


def test_both_providers_fail(monkeypatch):
    def fake_provider(provider, prompt, json_mode=False):
        return None, f"{provider} is down."

    monkeypatch.setattr(ai_analysis, "_call_provider", fake_provider)
    text, meta = ai_analysis.call_llm("Gemini", "prompt")

    assert text is None
    assert "Gemini" in meta["error"]
    assert "Groq" in meta["error"]
