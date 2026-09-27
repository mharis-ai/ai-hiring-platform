import json
import os

from google.genai import types


def extract_resume_pages(uploaded_file):
    """
    Prepare the candidate resume PDF as a Gemini PDF part.

    The original PDF is sent directly to Gemini instead of
    converting PDF pages into images. This avoids the need for
    Poppler and works better in cloud environments such as
    Streamlit Community Cloud.
    """

    if uploaded_file is None:
        raise ValueError("No resume file was provided.")

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        raise ValueError("The uploaded resume file is empty.")

    file_name = getattr(
        uploaded_file,
        "name",
        "resume.pdf"
    )

    if not file_name.lower().endswith(".pdf"):
        raise ValueError(
            "Please upload a valid PDF resume."
        )

    try:
        pdf_part = types.Part.from_bytes(
            data=file_bytes,
            mime_type="application/pdf"
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to prepare the resume PDF for Gemini."
        ) from exc

    return [pdf_part]


def clean_json_response(response_text):
    """
    Clean Gemini responses that may contain Markdown
    code fences before JSON parsing.
    """

    if not response_text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    response_text = response_text.strip()

    if response_text.startswith("```json"):
        response_text = response_text[7:]

    elif response_text.startswith("```JSON"):
        response_text = response_text[7:]

    elif response_text.startswith("```"):
        response_text = response_text[3:]

    if response_text.endswith("```"):
        response_text = response_text[:-3]

    return response_text.strip()


def validate_agent_output(analysis):
    """
    Ensure the AI Hiring Agent returns the expected structure.

    Missing fields are filled with safe defaults so the UI
    does not break if Gemini omits an optional field.
    """

    if not isinstance(analysis, dict):
        raise ValueError(
            "AI Hiring Agent returned an invalid JSON object."
        )

    default_structure = {
        "candidate_name": "Unknown Candidate",
        "professional_profile": "",
        "resume_alignment_score": 0,
        "matching_skills": [],
        "matching_keywords": [],
        "missing_or_unverified_requirements": [],
        "experience_alignment": {
            "summary": "",
            "relevant_experience": [],
            "areas_needing_clarification": []
        },
        "education_alignment": {
            "summary": "",
            "relevant_education": []
        },
        "project_alignment": {
            "summary": "",
            "relevant_projects": []
        },
        "candidate_strengths": [],
        "development_or_gap_areas": [],
        "interview_focus_areas": [],
        "suggested_interview_questions": [],
        "recruiter_notes": [],
        "verification_items": [],
        "human_review_notice": (
            "AI-generated analysis is provided for decision "
            "support only. Final recruitment decisions must be "
            "made by a qualified human recruiter after reviewing "
            "the original candidate information."
        )
    }

    for key, default_value in default_structure.items():
        if key not in analysis:
            analysis[key] = default_value

    try:
        score = float(
            analysis.get(
                "resume_alignment_score",
                0
            )
        )

        score = max(
            0,
            min(
                100,
                score
            )
        )

        if score.is_integer():
            score = int(score)

        analysis["resume_alignment_score"] = score

    except (TypeError, ValueError):
        analysis["resume_alignment_score"] = 0

    list_fields = [
        "matching_skills",
        "matching_keywords",
        "missing_or_unverified_requirements",
        "candidate_strengths",
        "development_or_gap_areas",
        "interview_focus_areas",
        "recruiter_notes",
        "verification_items"
    ]

    for field in list_fields:
        if not isinstance(
            analysis.get(field),
            list
        ):
            analysis[field] = []

    nested_fields = {
        "experience_alignment": [
            "summary",
            "relevant_experience",
            "areas_needing_clarification"
        ],
        "education_alignment": [
            "summary",
            "relevant_education"
        ],
        "project_alignment": [
            "summary",
            "relevant_projects"
        ]
    }

    for section, fields in nested_fields.items():

        if not isinstance(
            analysis.get(section),
            dict
        ):
            analysis[section] = {}

        for field in fields:

            if field not in analysis[section]:

                if field == "summary":
                    analysis[section][field] = ""

                else:
                    analysis[section][field] = []

    if not isinstance(
        analysis.get("suggested_interview_questions"),
        list
    ):
        analysis["suggested_interview_questions"] = []

    cleaned_questions = []

    for question in analysis[
        "suggested_interview_questions"
    ]:

        if isinstance(question, dict):

            cleaned_questions.append(
                {
                    "question": str(
                        question.get(
                            "question",
                            ""
                        )
                    ),
                    "reason": str(
                        question.get(
                            "reason",
                            ""
                        )
                    )
                }
            )

        elif isinstance(question, str):

            cleaned_questions.append(
                {
                    "question": question,
                    "reason": ""
                }
            )

    analysis["suggested_interview_questions"] = (
        cleaned_questions
    )

    return analysis


def run_hiring_agent(
    client,
    uploaded_file,
    job_description,
    knowledge=None
):
    """
    AI Hiring Agent.

    Analyzes a candidate resume against a job description
    and optionally uses recruiter knowledge extracted from
    approved recruitment documents.

    This agent provides decision-support information only.
    It does not make autonomous hiring decisions.
    """

    if client is None:
        raise ValueError(
            "Gemini client is not configured."
        )

    if not job_description or not job_description.strip():
        raise ValueError(
            "Please provide a job description."
        )

    # ---------------------------------------------------------
    # Resume processing
    # ---------------------------------------------------------

    resume_pages = extract_resume_pages(
        uploaded_file
    )

    # ---------------------------------------------------------
    # Recruitment knowledge
    # ---------------------------------------------------------

    if knowledge is None:
        knowledge = {}

    if not isinstance(knowledge, dict):
        knowledge = {
            "knowledge": knowledge
        }

    try:
        knowledge_json = json.dumps(
            knowledge,
            indent=2,
            ensure_ascii=False
        )
    except (TypeError, ValueError):
        knowledge_json = "{}"

    # ---------------------------------------------------------
    # Model configuration
    # ---------------------------------------------------------

    model_name = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.5-flash-lite"
    )

    # ---------------------------------------------------------
    # Main agent instructions
    # ---------------------------------------------------------

    prompt = f"""
You are the AI Hiring Agent inside an
AI Hiring & Career Intelligence Platform.

Your role is to provide structured,
evidence-based recruitment intelligence
for a HUMAN RECRUITER.

You are a DECISION-SUPPORT SYSTEM.

==================================================
CORE RULES
==================================================

You MUST:

- Analyze only information supported by the resume.
- Compare the resume against the provided job description.
- Use recruitment knowledge only when relevant.
- Clearly distinguish missing information from negative information.
- Identify information that requires human verification.
- Keep conclusions factual and evidence-based.
- Explain alignment using documented evidence.

You MUST NOT:

- Hire a candidate.
- Reject a candidate.
- Select a candidate.
- Recommend a final hiring decision.
- Predict job performance.
- Predict future success.
- Infer personality.
- Infer intelligence.
- Infer motivation.
- Infer protected characteristics.
- Invent experience.
- Invent education.
- Invent skills.
- Invent certifications.
- Invent projects.
- Invent achievements.
- Invent technologies.
- Treat missing information as proof that the candidate lacks it.

If something cannot be verified from the supplied material,
write that it is "not available" or "requires verification."

==================================================
ALIGNMENT SCORE
==================================================

You may provide:

"resume_alignment_score"

from 0 to 100.

This score measures ONLY the degree of documented
resume alignment with the provided job description.

It is NOT:

- a hiring score
- a candidate quality score
- an intelligence score
- a personality score
- a performance prediction
- a final recruitment recommendation

Do not convert this score into a hire/reject decision.

==================================================
ANALYSIS AREAS
==================================================

Analyze:

1. Candidate identity
2. Professional profile
3. Relevant skills
4. Technical skills
5. Job requirement alignment
6. Relevant experience
7. Education alignment
8. Project alignment
9. Matching keywords
10. Missing or unverified requirements
11. Evidence-based strengths
12. Development or gap areas
13. Areas requiring clarification
14. Interview focus areas
15. Candidate-specific interview questions
16. Recruiter notes
17. Verification items

==================================================
EVIDENCE RULE
==================================================

Every important observation should be based on:

A. Candidate resume
B. Job description
C. Recruitment knowledge

Do not introduce outside assumptions.

If the job requires Python and the resume
explicitly lists Python:

-> Python can be included as a matching skill.

If the job requires Docker but the resume
does not mention Docker:

-> Docker should be listed as missing or unverified.

Do NOT say:

"The candidate cannot use Docker."

Instead say:

"Docker is not explicitly documented in the resume
and should be verified if required for the role."

==================================================
CANDIDATE NAME
==================================================

Extract the candidate's name only from the resume.

If the name cannot be confidently identified:

"Unknown Candidate"

==================================================
PROFESSIONAL PROFILE
==================================================

Provide a concise factual summary based
only on the resume.

==================================================
MATCHING SKILLS
==================================================

List skills that:

- Are explicitly supported by the resume
- Are relevant to the job

Do not infer skills from job titles alone.

==================================================
MATCHING KEYWORDS
==================================================

Identify relevant job-description terms
that are actually supported by the resume.

Do not add keywords merely because
they appear in the job description.

==================================================
MISSING OR UNVERIFIED REQUIREMENTS
==================================================

List requirements that are:

- Not mentioned
- Unclear
- Partially documented
- Requiring verification

Do not treat absence of evidence as
proof of absence.

==================================================
EXPERIENCE ALIGNMENT
==================================================

Explain how documented experience relates
to the job requirements.

Do not invent responsibilities,
employers, dates, or achievements.

==================================================
EDUCATION ALIGNMENT
==================================================

Compare documented education with
the education requirements of the job.

==================================================
PROJECT ALIGNMENT
==================================================

Mention only projects that are actually
documented in the resume.

==================================================
CANDIDATE STRENGTHS
==================================================

List evidence-based strengths relevant
to the role.

Do not use vague personality judgments.

==================================================
DEVELOPMENT / GAP AREAS
==================================================

Identify documented gaps or areas that
require additional clarification.

Use neutral language.

==================================================
INTERVIEW FOCUS
==================================================

Suggest areas that a human recruiter
may want to explore.

Examples:

- Technical depth
- Practical implementation
- Project ownership
- API experience
- Automation workflows
- Communication of technical decisions
- Experience with required tools

Only include areas relevant to the
resume and job description.

==================================================
INTERVIEW QUESTIONS
==================================================

Generate candidate-specific questions
based on:

- Resume
- Job description
- Missing information
- Relevant skills
- Projects
- Experience

Each question must contain:

"question"

and

"reason"

Questions should help a human recruiter
verify information rather than automatically
judge the candidate.

==================================================
RECRUITER NOTES
==================================================

Provide concise, objective notes useful
during manual review.

==================================================
VERIFICATION ITEMS
==================================================

List information that should be verified
during recruitment.

Examples:

- Employment dates
- Claimed technical experience
- Certification validity
- Project ownership
- Specific tool experience
- Required skills not clearly demonstrated

Only include appropriate verification items.

==================================================
HUMAN REVIEW NOTICE
==================================================

Return exactly:

"AI-generated analysis is provided for decision
support only. Final recruitment decisions must
be made by a qualified human recruiter after
reviewing the original candidate information."

==================================================
OUTPUT FORMAT
==================================================

Return ONLY valid JSON.

Do not return Markdown.

Do not return ```json.

Use exactly this structure:

{{
    "candidate_name": "",

    "professional_profile": "",

    "resume_alignment_score": 0,

    "matching_skills": [],

    "matching_keywords": [],

    "missing_or_unverified_requirements": [],

    "experience_alignment": {{
        "summary": "",
        "relevant_experience": [],
        "areas_needing_clarification": []
    }},

    "education_alignment": {{
        "summary": "",
        "relevant_education": []
    }},

    "project_alignment": {{
        "summary": "",
        "relevant_projects": []
    }},

    "candidate_strengths": [],

    "development_or_gap_areas": [],

    "interview_focus_areas": [],

    "suggested_interview_questions": [
        {{
            "question": "",
            "reason": ""
        }}
    ],

    "recruiter_notes": [],

    "verification_items": [],

    "human_review_notice": ""
}}

==================================================
JOB DESCRIPTION
==================================================

{job_description}

==================================================
RECRUITMENT KNOWLEDGE
==================================================

{knowledge_json}

==================================================
FINAL INSTRUCTION
==================================================

Analyze the uploaded candidate resume against
the provided job description.

Use recruitment knowledge only when relevant.

Remain factual, evidence-based and neutral.

The final output is for human recruiter
decision support only.
"""

    # ---------------------------------------------------------
    # Gemini input
    # ---------------------------------------------------------

    input_text = f"""
Analyze the uploaded candidate resume for this role.

JOB DESCRIPTION:

{job_description}

Recruitment knowledge may contain company
policies, hiring requirements, processes,
or other approved information.

Use it only when relevant.

Remember:

- Use evidence from the supplied material.
- Do not invent information.
- Do not make hiring decisions.
- Do not recommend hire or reject.
- Provide decision-support information
  for a human recruiter.
"""

    contents = [
        input_text
    ]

    contents.extend(
        resume_pages
    )

    contents.append(
        prompt
    )

    # ---------------------------------------------------------
    # Gemini generation
    # ---------------------------------------------------------

    try:

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
        )

    except Exception as exc:

        raise RuntimeError(
            f"AI Hiring Agent failed to generate "
            f"the analysis: {exc}"
        ) from exc

    # ---------------------------------------------------------
    # Response handling
    # ---------------------------------------------------------

    response_text = getattr(
        response,
        "text",
        None
    )

    if not response_text:

        raise ValueError(
            "AI Hiring Agent received an empty response "
            "from Gemini."
        )

    response_text = clean_json_response(
        response_text
    )

    # ---------------------------------------------------------
    # JSON parsing
    # ---------------------------------------------------------

    try:

        analysis = json.loads(
            response_text
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            "AI Hiring Agent returned invalid JSON. "
            "The response could not be parsed."
        ) from exc

    # ---------------------------------------------------------
    # Validate and normalize
    # ---------------------------------------------------------

    analysis = validate_agent_output(
        analysis
    )

    return analysis
