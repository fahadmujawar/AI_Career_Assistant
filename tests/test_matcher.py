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
    # Punctuation is removed before phrases are built, so the keywords are:
    # python, docker, kubernetes, python docker, docker kubernetes.
    # The CV has 2 of the 5 -> 40.0.
    jd = "Python Docker. Kubernetes."
    result = match_keywords("Python developer. Docker user.", jd)

    assert sorted(result["matched"]) == ["docker", "python"]
    assert sorted(result["missing"]) == ["docker kubernetes", "kubernetes", "python docker"]
    assert result["score"] == 40.0
