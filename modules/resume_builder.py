from modules.gemini_utils import (
    create_pdf_part,
    generate_gemini_json,
)


def build_resume(
    client,
    uploaded_file,
    job_description
):

    pdf_part = create_pdf_part(
        uploaded_file
    )

    prompt = f"""
You are an AI Resume Builder Assistant.

Create a professional ATS-friendly resume structure
based ONLY on the candidate's actual resume.

IMPORTANT RULES:

1. Analyze the entire resume.
2. Analyze the entire job description.
3. NEVER invent information.
4. NEVER create fake skills, experience,
   education, projects, certifications,
   employers, achievements, or technologies.
5. Preserve real candidate information.
6. Improve wording and organization where appropriate.
7. Never fabricate numbers or achievements.
8. Return only JSON.

Return exactly:

{{
    "candidate_name": "",
    "professional_title": "",
    "contact_information": {{
        "email": "",
        "phone": "",
        "location": "",
        "linkedin": "",
        "github": ""
    }},
    "professional_summary": "",
    "skills": [],
    "work_experience": [
        {{
            "job_title": "",
            "company": "",
            "location": "",
            "dates": "",
            "responsibilities": []
        }}
    ],
    "projects": [
        {{
            "project_name": "",
            "description": "",
            "technologies": []
        }}
    ],
    "education": [
        {{
            "degree": "",
            "institution": "",
            "dates": ""
        }}
    ],
    "certifications": [],
    "additional_information": [],
    "truthfulness_warning": ""
}}

Professional summary must use only supported information.

Skills must be genuinely present.

Projects must actually exist in the resume.

Certifications must never be invented.

Contact information must only be extracted
if visible in the resume.

LinkedIn and GitHub must only be included
if they appear in the resume.

TARGET JOB DESCRIPTION:

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
