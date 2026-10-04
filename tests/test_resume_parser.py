import pytest

from utils.resume_parser import parse_resume

CV = """JANE DOE
jane.doe@example.com
+966 50 123 4567
SUMMARY
Data scientist with 5 years of experience.
EXPERIENCE
Built churn models in Python.
EDUCATION
BSc Computer Science
"""


def test_parse_resume_basic_fields():
    resume = parse_resume(CV)

    assert resume["name"] == "Jane Doe"
    assert resume["email"] == "jane.doe@example.com"
    assert resume["phone"].strip() == "+966 50 123 4567"


def test_parse_resume_sections():
    sections = parse_resume(CV)["sections"]

    assert sections["SUMMARY"] == ["Data scientist with 5 years of experience."]
    assert sections["EXPERIENCE"] == ["Built churn models in Python."]
    assert sections["EDUCATION"] == ["BSc Computer Science"]
    assert "jane.doe@example.com" in sections["HEADER"]


@pytest.mark.xfail(strict=True, reason="Known bug: phone regex also captures the line break after the number.")
def test_phone_has_no_trailing_line_break():
    assert parse_resume(CV)["phone"] == "+966 50 123 4567"


@pytest.mark.xfail(strict=True, reason="Known bug: a heading with a colon, like 'SKILLS:', is not detected.")
def test_heading_with_colon_is_detected():
    sections = parse_resume("JANE DOE\nSKILLS:\nPython, SQL\n")["sections"]
    assert sections.get("SKILLS") == ["Python, SQL"]
