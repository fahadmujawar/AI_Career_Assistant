import json
from utils.ai_client import get_gemini_client


def build_analysis_prompt(resume_sections, job_description):
    relevant_sections = {
        key: "\n".join(lines)
        for key, lines in resume_sections.items()
        if key != "HEADER"
    }

    resume_text = "\n\n".join(
        f"{section}:\n{content}"
        for section, content in relevant_sections.items()
    )

    prompt = f"""
You are an assistant helping a job seeker understand how well their CV matches a job description.

CV CONTENT:
{resume_text}

JOB DESCRIPTION:
{job_description}

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


def analyze_cv_against_jd(resume_sections, job_description):
    client = get_gemini_client()
    prompt = build_analysis_prompt(resume_sections, job_description)

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )

    try:
        result = json.loads(response.text)
    except json.JSONDecodeError:
        result = {
            "error": "Could not parse AI response as JSON.",
            "raw_response": response.text
        }

    return result