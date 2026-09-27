from modules.gemini_utils import (
    create_pdf_part,
    generate_gemini_json,
)


def optimize_resume(
    client,
    uploaded_file,
    job_description
):

    pdf_part = create_pdf_part(
        uploaded_file
    )

    prompt = f"""
You are an AI Resume Optimization Assistant.

Analyze the candidate's complete resume against
the provided job description.

Your purpose is to provide improvement suggestions
that help the candidate present their REAL qualifications
more clearly.

IMPORTANT RULES:

1. Analyze the entire resume.
2. Analyze the entire job description.
3. NEVER invent skills, experience, education,
   certifications, projects, employers, achievements,
   technologies, or qualifications.
4. Only recommend keywords genuinely supported by the resume.
5. Missing requirements must be clearly identified.
6. Do not fabricate experience.
7. Do not fabricate achievements or numbers.
8. Keep suggestions truthful and ATS-friendly.
9. Do not make hiring decisions.

Return exactly:

{{
    "optimization_score": 0,
    "resume_strengths": [],
    "keywords_to_emphasize": [],
    "missing_job_keywords": [],
    "skills_alignment": {{
        "matching_skills": [],
        "missing_skills": []
    }},
    "professional_summary": {{
        "current_assessment": "",
        "suggestion": ""
    }},
    "experience_improvements": [
        {{
            "original": "",
            "suggested": "",
            "reason": ""
        }}
    ],
    "project_improvements": [
        {{
            "project": "",
            "suggestion": ""
        }}
    ],
    "education_improvements": [],
    "ats_formatting_suggestions": [],
    "overall_suggestions": [],
    "important_warning": ""
}}

OPTIMIZATION SCORE:

90-100 = Highly Optimized
75-89 = Well Optimized
60-74 = Needs Improvement
0-59 = Needs Significant Improvement

This is NOT a hiring score.

The important_warning field must remind the user:

"Do not add any skill, experience, qualification,
or achievement that you do not genuinely have."

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