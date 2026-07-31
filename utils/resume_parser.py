import re

SECTION_ALIASES = {
    "SUMMARY": [
        "PROFESSIONAL SUMMARY", "SUMMARY", "PROFILE", "PROFESSIONAL PROFILE",
        "CAREER SUMMARY", "OBJECTIVE"
    ],
    "STRENGTHS": [
        "RELEVANT STRENGTHS", "KEY STRENGTHS", "CORE STRENGTHS"
    ],
    "SKILLS": [
        "TECHNICAL SKILLS", "SKILLS", "CORE SKILLS", "CORE COMPETENCIES", "KEY SKILLS"
    ],
    "PROJECTS": [
        "MACHINE LEARNING PROJECTS", "APPLIED AI & DATA PROJECTS", "PROJECTS",
        "KEY PROJECTS", "ACADEMIC PROJECTS", "RELEVANT PROJECTS"
    ],
    "RESEARCH": [
        "RESEARCH", "PUBLICATIONS", "RESEARCH & PUBLICATIONS"
    ],
    "EXPERIENCE": [
        "PROFESSIONAL EXPERIENCE", "WORK EXPERIENCE", "EXPERIENCE",
        "EMPLOYMENT HISTORY", "CAREER HISTORY"
    ],
    "EDUCATION": [
        "EDUCATION", "EDUCATION & QUALIFICATIONS", "ACADEMIC BACKGROUND"
    ],
    "CERTIFICATIONS": [
        "CERTIFICATIONS", "CERTIFICATES", "LICENSES & CERTIFICATIONS"
    ],
    "ADDITIONAL": [
        "ADDITIONAL INFORMATION", "ADDITIONAL", "OTHER INFORMATION", "INTERESTS"
    ],
}


def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else ""


def extract_phone(text):
    match = re.search(r'\+\d[\d\s-]{8,}', text)
    return match.group(0) if match else ""


def extract_name(text):
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line.title()
    return ""


def match_section_header(line):
    normalized = line.strip().upper()
    for category, aliases in SECTION_ALIASES.items():
        if normalized in aliases:
            return category
    return None


def split_sections(text):
    sections = {}
    current_category = "HEADER"
    sections[current_category] = []

    for line in text.splitlines():
        matched_category = match_section_header(line)

        if matched_category is not None:
            current_category = matched_category
            if current_category not in sections:
                sections[current_category] = []
        else:
            sections[current_category].append(line)

    return sections


def parse_resume(text):
    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "sections": split_sections(text),
    }