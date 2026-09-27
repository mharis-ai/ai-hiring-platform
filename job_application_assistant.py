from modules.gemini_utils import (
    create_pdf_part,
    generate_gemini_json,
)


def generate_application_package(
    client,
    uploaded_file,
    job_description
):

    pdf_part = create_pdf_part(
        uploaded_file
    )

    prompt = f"""
You are an AI Job Application Assistant.

Create a professional personalized application package
using ONLY the candidate's actual resume and target job description.

IMPORTANT RULES:

1. Analyze the complete resume.
2. Analyze the complete job description.
3. Never invent experience.
4. Never invent skills.
5. Never invent projects.
6. Never invent certifications.
7. Never invent employers.
8. Never invent achievements.
9. Never invent numbers or metrics.
10. Never claim unsupported qualifications.
11. Missing requirements must not be falsely claimed.
12. Do not make hiring decisions.
13. Return only JSON.

Return exactly:

{{
    "application_package": {{
        "cover_letter": "",
        "application_email": {{
            "subject": "",
            "body": ""
        }},
        "recruiter_message": "",
        "key_talking_points": []
    }},
    "job_alignment": {{
        "matching_qualifications": [],
        "missing_requirements": [],
        "relevant_keywords": []
    }},
    "truthfulness_warning": ""
}}

COVER LETTER:

Write a professional tailored cover letter.

APPLICATION EMAIL:

Create a concise professional application email.

Do not invent the recruiter's name.

RECRUITER MESSAGE:

Create a short natural recruiter/LinkedIn message.

KEY TALKING POINTS:

Create 4-7 concise genuine talking points.

JOB ALIGNMENT:

Only identify qualifications actually supported
by the resume.

TRUTHFULNESS WARNING:

"Review all generated content carefully.
Do not claim any skill, experience,
qualification, certification, achievement,
or responsibility that you do not genuinely have."

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