import pytest

from utils.matcher import get_important_keywords, match_keywords


def test_nosql_and_rapid_do_not_match_sql_or_api():
    jd = "Strong SQL skills. Build an API. SQL and API design."
    result = match_keywords("rapid prototype in NoSQL", jd)

    assert "sql" not in result["matched"]
    assert "api" not in result["matched"]


def test_short_keywords_ai_and_ml_are_found():
    jd = "AI engineer with ML experience. AI and ML in production."
    assert "ai" in get_important_keywords(jd)
    assert "ml" in get_important_keywords(jd)

    result = match_keywords("Built AI and ML systems", jd)
    assert "ai" in result["matched"]
    assert "ml" in result["matched"]


def test_no_doubled_phrases():
    keywords = get_important_keywords("SQL SQL sql. Python python.")

    for keyword in keywords:
        words = keyword.split()
        assert not (len(words) == 2 and words[0] == words[1]), keyword


def test_two_word_phrase_and_whole_word_match():
    jd = "Machine learning engineer. Python and machine learning."
    result = match_keywords("I use Python for machine learning.", jd)

    assert "machine learning" in result["matched"]
    assert "python" in result["matched"]


def test_exact_score():
    # Keywords: python, sql, docker, python sql, sql docker.
    # The CV has python, sql and "python sql" -> 3 of 5 -> 60.0.
    jd = "Python SQL Docker"
    result = match_keywords("I use Python SQL daily", jd)

    assert sorted(result["matched"]) == ["python", "python sql", "sql"]
    assert sorted(result["missing"]) == ["docker", "sql docker"]
    assert result["score"] == 60.0


@pytest.mark.xfail(strict=True, reason="word pairs join across sentence ends")
def test_word_pairs_do_not_cross_sentence_ends():
    assert "docker kubernetes" not in get_important_keywords("Docker. Kubernetes")
