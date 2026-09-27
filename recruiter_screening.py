from modules.gemini_utils import (
    create_pdf_part,
    generate_gemini_json,
)


def screen_candidate(
    client,
    uploaded_file,
    job_description
):

    pdf_part = create_pdf_part(
        uploaded_file
    )

    prompt = f"""
You are an AI recruitment analysis assistant.

Analyze the candidate's complete resume against
the provided job description.

IMPORTANT RULES:

1. Analyze the entire resume.
2. Analyze the entire job description.
3. Use only information actually demonstrated in the resume.
4. Never invent skills, experience, education, projects,
   certifications, employers, achievements, or technologies.
5. Do not make hiring decisions.
6. Provide decision-support information for a human recruiter.
7. ATS score must be a single integer from 0 to 100.
8. Base the score only on job-relevant evidence.

Evaluate:

- Relevant keywords
- Technical skills
- Relevant experience
- Education
- Projects
- Job responsibilities
- Required technologies
- Important missing requirements

Return exactly this JSON:

{{
    "candidate_name": "",
    "ats_match_score": 0,
    "match_level": "",
    "matching_skills": [],
    "missing_skills": [],
    "matching_keywords": [],
    "missing_keywords": [],
    "experience_match": "",
    "education_match": "",
    "strengths": [],
    "gaps": [],
    "important_missing_requirements": [],
    "recruiter_summary": ""
}}

MATCH LEVEL:

80-100 = Strong Match
60-79 = Moderate Match
40-59 = Limited Match
0-39 = Low Match

The candidate_name must only be taken from the resume.

If the candidate's name cannot be identified,
return "Unknown Candidate".

The recruiter_summary must be short and objective.

Do not say Hire, Reject, Select,
or make the final hiring decision.

JOB DESCRIPTION:

{job_description}
"""

    contents = [
        pdf_part,
        prompt
    ]

    return generate_gemini_json(
        client,
        contents
    )