from modules.gemini_utils import (
    create_pdf_part,
    generate_gemini_json,
)


def generate_interview_preparation(
    client,
    uploaded_file,
    job_description
):

    pdf_part = create_pdf_part(
        uploaded_file
    )

    prompt = f"""
You are an AI Interview Preparation Assistant.

Prepare the candidate for an interview using ONLY
their actual resume and the target job description.

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
10. Questions may cover job requirements not present
    in the resume.
11. Clearly identify unsupported requirements.
12. Model answer guidance must remain truthful.
13. Do not create fake personal experiences.
14. Do not make hiring decisions.
15. Do not predict interview results.
16. Return only JSON.

Return exactly:

{{
    "interview_overview": {{
        "role_focus": "",
        "key_topics": [],
        "preparation_summary": ""
    }},

    "technical_questions": [
        {{
            "question": "",
            "why_it_may_be_asked": "",
            "answer_guidance": "",
            "resume_support": "",
            "difficulty": ""
        }}
    ],

    "behavioral_questions": [
        {{
            "question": "",
            "why_it_may_be_asked": "",
            "answer_guidance": ""
        }}
    ],

    "resume_based_questions": [
        {{
            "question": "",
            "resume_reference": "",
            "answer_guidance": ""
        }}
    ],

    "job_specific_questions": [
        {{
            "question": "",
            "job_requirement": "",
            "answer_guidance": ""
        }}
    ],

    "potential_weak_areas": [
        {{
            "area": "",
            "reason": "",
            "preparation_tip": ""
        }}
    ],

    "preparation_checklist": []
}}

Generate approximately:

Technical questions: 8-12
Behavioral questions: 5-7
Resume-based questions: 5-7
Job-specific questions: 5-7
Checklist: 10-15 items

Answer guidance must encourage the candidate
to use their real experiences.

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