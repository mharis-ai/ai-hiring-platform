import io
import os
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from google import genai

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER

from modules.recruiter_screening import screen_candidate
from modules.resume_optimizer import optimize_resume
from modules.resume_builder import build_resume
from modules.job_application_assistant import (
    generate_application_package
)
from modules.interview_assistant import (
    generate_interview_preparation
)
from modules.rag_assistant import (
    analyze_multiple_documents,
    ask_knowledge_assistant,
)
from modules.hiring_agent import (
    run_hiring_agent,
)

from modules.gemini_utils import (
    format_gemini_error,
)

from modules.auth import (
    initialize_auth_state,
    get_current_user,
    get_user_name,
    get_user_role,
    sign_in,
    sign_up,
    request_password_reset,
    sign_out,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Hiring Platform",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# AUTHENTICATION GATE
# ============================================================

def render_auth_page():

    st.markdown(
        """
        <style>
        .auth-wrap { max-width: 520px; margin: 60px auto 0 auto; }
        .auth-title { font-size: 34px; font-weight: 800; text-align: center; }
        .auth-subtitle { text-align: center; color: #6b7280; margin-bottom: 24px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">🤖 AI Hiring Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="auth-subtitle">Sign in to continue to your hiring workspace.</div>',
        unsafe_allow_html=True,
    )

    login_tab, signup_tab, forgot_tab = st.tabs(
        ["🔐 Login", "📝 Sign Up", "🔑 Forgot Password"]
    )

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "Login",
                use_container_width=True,
                type="primary",
            )

        if submitted:
            success, message, _ = sign_in(email, password)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    with signup_tab:
        with st.form("signup_form"):
            full_name = st.text_input("Full Name", placeholder="Your full name")
            email = st.text_input("Email", placeholder="you@example.com", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                key="signup_confirm_password",
            )
            role = st.selectbox("Account Type", ["Recruiter", "Job Seeker"])
            submitted = st.form_submit_button(
                "Create Account",
                use_container_width=True,
                type="primary",
            )

        if submitted:
            if password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message, _ = sign_up(full_name, email, password, role)
                if success:
                    st.success(message)
                else:
                    st.error(message)

    with forgot_tab:
        st.caption("Enter your account email and we will send a password reset link.")
        with st.form("forgot_password_form"):
            email = st.text_input("Email", placeholder="you@example.com", key="forgot_email")
            submitted = st.form_submit_button(
                "Send Reset Link",
                use_container_width=True,
                type="primary",
            )

        if submitted:
            success, message = request_password_reset(email)
            if success:
                st.success(message)
            else:
                st.error(message)

    st.markdown('</div>', unsafe_allow_html=True)


initialize_auth_state()
current_user = get_current_user()

if not current_user:
    render_auth_page()
    st.stop()


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY"
)

if not GOOGLE_API_KEY:

    st.error(
        "GOOGLE_API_KEY is missing.\n\n"
        "Please add your Gemini API key "
        "to the .env file."
    )

    st.stop()


try:

    client = genai.Client(
        api_key=GOOGLE_API_KEY
    )

except Exception as error:

    st.error(
        f"Unable to initialize Gemini client.\n\n"
        f"{error}"
    )

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_SESSION_STATE = {

    "optimizer_result": None,

    "builder_result": None,

    "application_result": None,

    "interview_result": None,

    "recruiter_results": [],

    "saved_jobs": [],

    "selected_job_index": None,

    "candidate_records": [],

    "followup_result": None,

    "rag_knowledge": None,

    "rag_document_name": None,

    "rag_answer": None,

    "hiring_agent_result": None,

    "hiring_agent_saved_message": None,
}


for key, value in DEFAULT_SESSION_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_list(value):

    if isinstance(value, list):
        return value

    if value is None:
        return []

    return [value]


def generate_candidate_id():

    return (
        "CAND-"
        + uuid.uuid4().hex[:8].upper()
    )


def get_job_key(
    job_title,
    company
):

    return (
        f"{str(job_title).strip()}|"
        f"{str(company).strip()}"
    ).lower()


def get_now_string():

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M"
    )


def normalize_text(value):

    return (
        str(value or "")
        .strip()
        .lower()
    )


def normalize_candidate_record(
    candidate
):

    candidate.setdefault(
        "candidate_id",
        generate_candidate_id()
    )

    candidate.setdefault(
        "candidate_name",
        "Unknown Candidate"
    )

    candidate.setdefault(
        "file_name",
        ""
    )

    candidate.setdefault(
        "job_title",
        ""
    )

    candidate.setdefault(
        "company",
        ""
    )

    candidate.setdefault(
        "job_key",
        get_job_key(
            candidate["job_title"],
            candidate["company"]
        )
    )

    candidate.setdefault(
        "ats_match_score",
        0
    )

    candidate.setdefault(
        "match_level",
        ""
    )

    candidate.setdefault(
        "matching_skills",
        []
    )

    candidate.setdefault(
        "missing_skills",
        []
    )

    candidate.setdefault(
        "strengths",
        []
    )

    candidate.setdefault(
        "gaps",
        []
    )

    candidate.setdefault(
        "recruiter_summary",
        ""
    )

    candidate.setdefault(
        "status",
        "New"
    )

    candidate.setdefault(
        "notes",
        ""
    )

    candidate.setdefault(
        "tags",
        []
    )

    candidate.setdefault(
        "interview",
        {
            "date": "",
            "time": "",
            "mode": "",
            "interviewer": "",
            "feedback": "",
            "rating": 0,
        }
    )

    candidate.setdefault(
        "next_action",
        ""
    )

    candidate.setdefault(
        "next_action_date",
        ""
    )

    candidate.setdefault(
        "screening_history",
        []
    )

    # --------------------------------------------------------
    # AI HIRING AGENT FIELDS
    # --------------------------------------------------------

    candidate.setdefault(
        "hiring_agent_analysis",
        None
    )

    candidate.setdefault(
        "hiring_agent_history",
        []
    )

    return candidate


def normalize_all_candidates():

    normalized = []

    for candidate in st.session_state.candidate_records:

        normalized.append(
            normalize_candidate_record(
                candidate
            )
        )

    st.session_state.candidate_records = normalized


def find_candidate_by_id(
    candidate_id
):

    for candidate in st.session_state.candidate_records:

        if candidate.get(
            "candidate_id"
        ) == candidate_id:

            return candidate

    return None


def find_existing_candidate(
    uploaded_file,
    job_title,
    company,
    candidate_name=""
):

    job_key = get_job_key(
        job_title,
        company
    )

    normalized_file_name = normalize_text(
        uploaded_file.name
    )

    normalized_candidate_name = normalize_text(
        candidate_name
    )

    for candidate in st.session_state.candidate_records:

        existing_file = normalize_text(
            candidate.get(
                "file_name",
                ""
            )
        )

        existing_job_key = candidate.get(
            "job_key",
            get_job_key(
                candidate.get(
                    "job_title",
                    ""
                ),
                candidate.get(
                    "company",
                    ""
                )
            )
        )

        if (
            existing_file
            == normalized_file_name
            and existing_job_key
            == job_key
        ):

            return candidate

    if normalized_candidate_name:

        for candidate in st.session_state.candidate_records:

            existing_name = normalize_text(
                candidate.get(
                    "candidate_name",
                    ""
                )
            )

            existing_job_key = candidate.get(
                "job_key",
                get_job_key(
                    candidate.get(
                        "job_title",
                        ""
                    ),
                    candidate.get(
                        "company",
                        ""
                    )
                )
            )

            if (
                existing_name
                and existing_name
                == normalized_candidate_name
                and existing_job_key
                == job_key
            ):

                return candidate

    return None


# ============================================================
# AI HIRING AGENT → CANDIDATE WORKFLOW
# ============================================================

def save_hiring_agent_to_candidate(
    result,
    uploaded_file,
    job_title,
    company
):

    """
    Save AI Hiring Agent analysis into the candidate
    workflow without overwriting the existing ATS screening
    score and without creating duplicate candidate records.
    """

    candidate_name = result.get(
        "candidate_name",
        "Unknown Candidate"
    )

    if not candidate_name:

        candidate_name = "Unknown Candidate"

    job_title = str(
        job_title or ""
    ).strip()

    company = str(
        company or ""
    ).strip()

    job_key = get_job_key(
        job_title,
        company
    )

    existing = find_existing_candidate(
        uploaded_file,
        job_title,
        company,
        candidate_name
    )

    # --------------------------------------------------------
    # CREATE HISTORY ENTRY
    # --------------------------------------------------------

    history_entry = {

        "timestamp":
            get_now_string(),

        "job_title":
            job_title,

        "company":
            company,

        "resume_file":
            uploaded_file.name,

        "resume_alignment_score":
            result.get(
                "resume_alignment_score",
                0
            ),
    }

    # --------------------------------------------------------
    # EXISTING CANDIDATE
    # --------------------------------------------------------

    if existing:

        existing["hiring_agent_analysis"] = result

        existing.setdefault(
            "hiring_agent_history",
            []
        )

        existing[
            "hiring_agent_history"
        ].append(
            history_entry
        )

        # Keep core candidate information synchronized.
        existing["candidate_name"] = candidate_name

        existing["file_name"] = uploaded_file.name

        existing["job_title"] = job_title

        existing["company"] = company

        existing["job_key"] = job_key

        # IMPORTANT:
        # Do NOT overwrite:
        # existing["ats_match_score"]
        #
        # AI Hiring Agent score is a separate analysis score.

        return existing, True

    # --------------------------------------------------------
    # NEW CANDIDATE FROM AI AGENT
    # --------------------------------------------------------

    agent_score = result.get(
        "resume_alignment_score",
        0
    )

    try:

        agent_score = int(
            agent_score
        )

    except Exception:

        agent_score = 0

    agent_score = max(
        0,
        min(
            100,
            agent_score
        )
    )

    candidate = {

        "candidate_id":
            generate_candidate_id(),

        "candidate_name":
            candidate_name,

        "file_name":
            uploaded_file.name,

        "job_title":
            job_title,

        "company":
            company,

        "job_key":
            job_key,

        # Screening score is kept separate.
        "ats_match_score":
            0,

        "match_level":
            "",

        "matching_skills":
            [],

        "missing_skills":
            [],

        "strengths":
            [],

        "gaps":
            [],

        "recruiter_summary":
            "",

        # Human-controlled workflow state.
        "status":
            "New",

        "notes":
            "",

        "tags":
            [],

        "interview": {

            "date":
                "",

            "time":
                "",

            "mode":
                "",

            "interviewer":
                "",

            "feedback":
                "",

            "rating":
                0,
        },

        "next_action":
            "",

        "next_action_date":
            "",

        "screening_history":
            [],

        # AI Hiring Agent data.
        "hiring_agent_analysis":
            result,

        "hiring_agent_history": [
            history_entry
        ],
    }

    st.session_state.candidate_records.append(
        candidate
    )

    return candidate, False


# ============================================================
# CREATE CANDIDATE FROM SCREENING
# ============================================================

def create_candidate_from_screening(
    result,
    uploaded_file,
    job_title,
    company
):

    candidate_name = result.get(
        "candidate_name",
        "Unknown Candidate"
    )

    score = result.get(
        "ats_match_score",
        0
    )

    try:

        score = int(score)

    except Exception:

        score = 0

    score = max(
        0,
        min(
            100,
            score
        )
    )

    job_key = get_job_key(
        job_title,
        company
    )

    existing = find_existing_candidate(
        uploaded_file,
        job_title,
        company,
        candidate_name
    )

    screening_entry = {

        "timestamp":
            get_now_string(),

        "job_title":
            job_title,

        "company":
            company,

        "score":
            score,

        "match_level":
            result.get(
                "match_level",
                ""
            ),
    }

    if existing:

        existing.update({

            "candidate_name":
                candidate_name,

            "file_name":
                uploaded_file.name,

            "job_title":
                job_title,

            "company":
                company,

            "job_key":
                job_key,

            "ats_match_score":
                score,

            "match_level":
                result.get(
                    "match_level",
                    ""
                ),

            "matching_skills":
                safe_list(
                    result.get(
                        "matching_skills"
                    )
                ),

            "missing_skills":
                safe_list(
                    result.get(
                        "missing_skills"
                    )
                ),

            "strengths":
                safe_list(
                    result.get(
                        "strengths"
                    )
                ),

            "gaps":
                safe_list(
                    result.get(
                        "gaps"
                    )
                ),

            "recruiter_summary":
                result.get(
                    "recruiter_summary",
                    ""
                ),
        })

        existing.setdefault(
            "screening_history",
            []
        )

        existing[
            "screening_history"
        ].append(
            screening_entry
        )

        return existing, True

    candidate = {

        "candidate_id":
            generate_candidate_id(),

        "candidate_name":
            candidate_name,

        "file_name":
            uploaded_file.name,

        "job_title":
            job_title,

        "company":
            company,

        "job_key":
            job_key,

        "ats_match_score":
            score,

        "match_level":
            result.get(
                "match_level",
                ""
            ),

        "matching_skills":
            safe_list(
                result.get(
                    "matching_skills"
                )
            ),

        "missing_skills":
            safe_list(
                result.get(
                    "missing_skills"
                )
            ),

        "strengths":
            safe_list(
                result.get(
                    "strengths"
                )
            ),

        "gaps":
            safe_list(
                result.get(
                    "gaps"
                )
            ),

        "recruiter_summary":
            result.get(
                "recruiter_summary",
                ""
            ),

        "status":
            "New",

        "notes":
            "",

        "tags":
            [],

        "interview": {

            "date":
                "",

            "time":
                "",

            "mode":
                "",

            "interviewer":
                "",

            "feedback":
                "",

            "rating":
                0,
        },

        "next_action":
            "",

        "next_action_date":
            "",

        "screening_history": [
            screening_entry
        ],

        "hiring_agent_analysis":
            None,

        "hiring_agent_history":
            [],
    }

    st.session_state.candidate_records.append(
        candidate
    )

    return candidate, False


# ============================================================
# CANDIDATE DATAFRAME
# ============================================================

def build_candidate_dataframe():

    rows = []

    for candidate in st.session_state.candidate_records:

        interview = candidate.get(
            "interview",
            {}
        )

        hiring_agent = candidate.get(
            "hiring_agent_analysis",
            {}
        ) or {}

        agent_score = hiring_agent.get(
            "resume_alignment_score",
            ""
        )

        rows.append({

            "Candidate ID":
                candidate.get(
                    "candidate_id",
                    ""
                ),

            "Candidate Name":
                candidate.get(
                    "candidate_name",
                    ""
                ),

            "Resume":
                candidate.get(
                    "file_name",
                    ""
                ),

            "Job":
                candidate.get(
                    "job_title",
                    ""
                ),

            "Company":
                candidate.get(
                    "company",
                    ""
                ),

            "ATS Score":
                candidate.get(
                    "ats_match_score",
                    0
                ),

            "AI Agent Score":
                agent_score,

            "Match Level":
                candidate.get(
                    "match_level",
                    ""
                ),

            "Status":
                candidate.get(
                    "status",
                    ""
                ),

            "Interview Date":
                interview.get(
                    "date",
                    ""
                ),

            "Interview Time":
                interview.get(
                    "time",
                    ""
                ),

            "Interview Mode":
                interview.get(
                    "mode",
                    ""
                ),

            "Interviewer":
                interview.get(
                    "interviewer",
                    ""
                ),

            "Interview Rating":
                interview.get(
                    "rating",
                    0
                ),

            "Next Action":
                candidate.get(
                    "next_action",
                    ""
                ),

            "Next Action Date":
                candidate.get(
                    "next_action_date",
                    ""
                ),

            "Tags":
                ", ".join(
                    safe_list(
                        candidate.get(
                            "tags",
                            []
                        )
                    )
                ),

            "Notes":
                candidate.get(
                    "notes",
                    ""
                ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# RESUME PDF CREATOR
# ============================================================

def create_resume_pdf(
    resume_data
):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]

    story = []

    candidate_name = resume_data.get(
        "candidate_name",
        "Resume"
    )

    professional_title = resume_data.get(
        "professional_title",
        ""
    )

    story.append(
        Paragraph(
            candidate_name,
            title_style
        )
    )

    if professional_title:

        story.append(
            Paragraph(
                professional_title,
                body_style
            )
        )

    story.append(
        Spacer(
            1,
            10
        )
    )

    contact = resume_data.get(
        "contact_information",
        {}
    )

    contact_items = []

    for key in [
        "email",
        "phone",
        "location",
        "linkedin",
        "github"
    ]:

        value = contact.get(
            key,
            ""
        )

        if value:

            contact_items.append(
                str(value)
            )

    if contact_items:

        story.append(
            Paragraph(
                " | ".join(
                    contact_items
                ),
                body_style
            )
        )

        story.append(
            Spacer(
                1,
                12
            )
        )

    summary = resume_data.get(
        "professional_summary",
        ""
    )

    if summary:

        story.append(
            Paragraph(
                "Professional Summary",
                heading_style
            )
        )

        story.append(
            Paragraph(
                summary,
                body_style
            )
        )

        story.append(
            Spacer(
                1,
                8
            )
        )

    skills = safe_list(
        resume_data.get(
            "skills",
            []
        )
    )

    if skills:

        story.append(
            Paragraph(
                "Skills",
                heading_style
            )
        )

        story.append(
            Paragraph(
                ", ".join(
                    map(
                        str,
                        skills
                    )
                ),
                body_style
            )
        )

        story.append(
            Spacer(
                1,
                8
            )
        )

    work_experience = safe_list(
        resume_data.get(
            "work_experience",
            []
        )
    )

    if work_experience:

        story.append(
            Paragraph(
                "Work Experience",
                heading_style
            )
        )

        for job in work_experience:

            title = job.get(
                "job_title",
                ""
            )

            company = job.get(
                "company",
                ""
            )

            dates = job.get(
                "dates",
                ""
            )

            story.append(
                Paragraph(
                    f"<b>{title}</b> — "
                    f"{company} "
                    f"{dates}",
                    body_style
                )
            )

            for responsibility in safe_list(
                job.get(
                    "responsibilities",
                    []
                )
            ):

                story.append(
                    Paragraph(
                        f"• {responsibility}",
                        body_style
                    )
                )

            story.append(
                Spacer(
                    1,
                    6
                )
            )

    projects = safe_list(
        resume_data.get(
            "projects",
            []
        )
    )

    if projects:

        story.append(
            Paragraph(
                "Projects",
                heading_style
            )
        )

        for project in projects:

            project_name = project.get(
                "project_name",
                ""
            )

            description = project.get(
                "description",
                ""
            )

            technologies = safe_list(
                project.get(
                    "technologies",
                    []
                )
            )

            story.append(
                Paragraph(
                    f"<b>{project_name}</b>",
                    body_style
                )
            )

            if description:

                story.append(
                    Paragraph(
                        description,
                        body_style
                    )
                )

            if technologies:

                story.append(
                    Paragraph(
                        "Technologies: "
                        + ", ".join(
                            map(
                                str,
                                technologies
                            )
                        ),
                        body_style
                    )
                )

            story.append(
                Spacer(
                    1,
                    6
                )
            )

    education = safe_list(
        resume_data.get(
            "education",
            []
        )
    )

    if education:

        story.append(
            Paragraph(
                "Education",
                heading_style
            )
        )

        for item in education:

            story.append(
                Paragraph(
                    f"<b>{item.get('degree', '')}</b> — "
                    f"{item.get('institution', '')} "
                    f"{item.get('dates', '')}",
                    body_style
                )
            )

    certifications = safe_list(
        resume_data.get(
            "certifications",
            []
        )
    )

    if certifications:

        story.append(
            Paragraph(
                "Certifications",
                heading_style
            )
        )

        for certification in certifications:

            story.append(
                Paragraph(
                    f"• {certification}",
                    body_style
                )
            )

    additional = safe_list(
        resume_data.get(
            "additional_information",
            []
        )
    )

    if additional:

        story.append(
            Paragraph(
                "Additional Information",
                heading_style
            )
        )

        for item in additional:

            story.append(
                Paragraph(
                    f"• {item}",
                    body_style
                )
            )

    story.append(
        Spacer(
            1,
            15
        )
    )

    story.append(
        HRFlowable(
            width="100%"
        )
    )

    story.append(
        Spacer(
            1,
            8
        )
    )

    story.append(
        Paragraph(
            "Please verify all information before using this resume.",
            body_style
        )
    )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# FOLLOW-UP MESSAGE GENERATOR
# ============================================================

def generate_followup_message(
    candidate,
    purpose
):

    name = candidate.get(
        "candidate_name",
        "Candidate"
    )

    job = candidate.get(
        "job_title",
        "the position"
    )

    company = candidate.get(
        "company",
        ""
    )

    company_phrase = (
        f" at {company}"
        if company.strip()
        else ""
    )

    if purpose == "Interview Invitation":

        return (
            f"Hello {name},\n\n"
            f"Thank you for your interest in the "
            f"{job} position{company_phrase}.\n\n"
            f"We would like to invite you to an interview "
            f"to discuss your background and experience "
            f"in more detail.\n\n"
            f"Please share your availability so we can "
            f"schedule a suitable time.\n\n"
            f"Best regards,\n"
            f"Recruitment Team"
        )

    if purpose == "Interview Follow-up":

        return (
            f"Hello {name},\n\n"
            f"Thank you for taking the time to interview "
            f"for the {job} position{company_phrase}.\n\n"
            f"We appreciate your time and interest. "
            f"Our team is currently reviewing the "
            f"interview feedback and we will keep you "
            f"updated regarding the next steps.\n\n"
            f"Best regards,\n"
            f"Recruitment Team"
        )

    if purpose == "Application Status Update":

        return (
            f"Hello {name},\n\n"
            f"We wanted to provide you with an update "
            f"regarding your application for the {job} "
            f"position{company_phrase}.\n\n"
            f"Your application is currently being reviewed. "
            f"We will share further information as the "
            f"recruitment process progresses.\n\n"
            f"Thank you for your patience.\n\n"
            f"Best regards,\n"
            f"Recruitment Team"
        )

    if purpose == "On Hold Update":

        return (
            f"Hello {name},\n\n"
            f"We are writing to provide an update regarding "
            f"your application for the {job} position"
            f"{company_phrase}.\n\n"
            f"Your application is currently on hold while "
            f"we complete the next stage of our recruitment "
            f"process.\n\n"
            f"We appreciate your patience and will contact "
            f"you when there is a further update.\n\n"
            f"Best regards,\n"
            f"Recruitment Team"
        )

    if purpose == "Next Steps":

        return (
            f"Hello {name},\n\n"
            f"We are pleased to provide an update regarding "
            f"your application for the {job} position"
            f"{company_phrase}.\n\n"
            f"We would like to proceed with the next step "
            f"of the recruitment process. Our team will "
            f"contact you with the relevant details and "
            f"schedule.\n\n"
            f"Best regards,\n"
            f"Recruitment Team"
        )

    return (
        f"Hello {name},\n\n"
        f"We wanted to provide you with an update "
        f"regarding your application for the {job} "
        f"position{company_phrase}.\n\n"
        f"We will share further information as the "
        f"process progresses.\n\n"
        f"Best regards,\n"
        f"Recruitment Team"
    )


# ============================================================
# NORMALIZE
# ============================================================

normalize_all_candidates()


# ============================================================
# MODERN DASHBOARD UI
# ============================================================

st.markdown(
    """
<style>
/* ---------- App shell ---------- */
[data-testid="stAppViewContainer"] {
    background: #f7f9fc;
}
[data-testid="stHeader"] {
    background: transparent;
}
section[data-testid="stSidebar"] {
    background: #101d34;
    border-right: 1px solid #1d2b45;
}
section[data-testid="stSidebar"] > div {
    background: #101d34;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #dce6f7 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #2a3953;
}

/* ---------- Sidebar ---------- */
.brand {
    padding: 6px 6px 18px 6px;
}
.brand-row {
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-icon {
    width: 44px;
    height: 44px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #d8e7ff, #7db2ff);
    color: #1e5bd7;
    font-size: 25px;
    font-weight: 800;
}
.brand-name {
    color: #ffffff;
    font-size: 20px;
    font-weight: 750;
    line-height: 1.1;
}
.brand-sub {
    color: #9fb0ca;
    font-size: 12px;
    margin-top: 4px;
}
.nav-label {
    color: #8fa2bf;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 700;
    margin: 18px 6px 8px;
}
.sidebar-profile {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 8px 2px;
}
.avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: #284b9b;
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
}
.profile-name { color: #fff; font-weight: 650; font-size: 13px; }
.profile-role { color: #8fa2bf; font-size: 11px; }

/* radio-as-navigation */
section[data-testid="stSidebar"] .stButton {
    margin: 0 0 3px 0;
}
section[data-testid="stSidebar"] .stButton > button {
    min-height: 30px;
    border-radius: 9px;
    border: 1px solid transparent;
    color: #dce6f7 !important;
    background: transparent !important;
    text-align: left;
    justify-content: flex-start;
    padding: 3px 11px;
    font-size: 12px;
    font-weight: 550;
    box-shadow: none !important;
    transition: background .15s ease, transform .15s ease;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #1b2b49 !important;
    border-color: transparent !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #245ecb, #2d70e8) !important;
    color: #ffffff !important;
    font-weight: 650;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: linear-gradient(90deg, #245ecb, #2d70e8) !important;
}
section[data-testid="stSidebar"] .stButton > button p {
    color: inherit !important;
}
[data-testid="stAppViewContainer"] .main {
    margin-left: 242px !important;
    width: calc(100% - 242px) !important;
}

section[data-testid="stSidebar"] {
    width: 242px !important;
    min-width: 242px !important;
    max-width: 242px !important;
}
section[data-testid="stSidebar"] > div {
    width: 242px !important;
}

/* ---------- Screenshot dashboard layout ---------- */
.dashboard-shell { max-width: 100%; }
.dashboard-hero {
    border: 1px solid #dfe6f3; border-radius: 12px; padding: 17px 20px;
    min-height: 126px; background: linear-gradient(105deg,#eef4ff 0%,#f3efff 100%);
    display:flex; align-items:center; gap:14px;
}
.dashboard-hero-title { color:#17213a; font-size:23px; font-weight:800; }
.dashboard-hero-copy { color:#65738c; font-size:12px; margin-top:4px; }
.dashboard-hero-date { color:#7a86a0; font-size:10px; margin-left:auto; align-self:flex-start; }
.dashboard-robot { font-size:46px; }
.quick-card,.rail-card,.dash-card { background:#fff; border:1px solid #e3e9f3; border-radius:12px; box-shadow:0 2px 8px rgba(15,23,42,.035); }
.quick-card { padding:12px 14px; }
.quick-title { color:#17213a; font-size:14px; font-weight:800; margin-bottom:8px; }
.quick-btn { border:1px solid #dbe4f5; border-radius:8px; padding:10px 11px; color:#20304e; font-size:11px; font-weight:700; margin-bottom:8px; background:#fbfcff; }
.dash-card { padding:13px; }
.dash-card-title { color:#17213a; font-size:14px; font-weight:800; }
.dash-muted { color:#71809a; font-size:10px; }
.metric-card { min-height:92px; }
.funnel-wrap { margin-top:10px; }
.funnel-row { margin:0 auto 7px; height:31px; border-radius:7px; display:flex; align-items:center; justify-content:space-between; padding:0 12px; color:#1e2b45; font-size:10px; font-weight:700; }
.funnel-row span:last-child { color:#58709b; }
.trend-box { height:220px; }
.trend-svg { width:100%; height:190px; display:block; }
.candidate-table { width:100%; border-collapse:collapse; margin-top:8px; }
.candidate-table th { text-align:left; color:#71809a; font-size:9px; font-weight:700; padding:7px 6px; border-bottom:1px solid #edf1f6; }
.candidate-table td { color:#26344d; font-size:9px; padding:8px 6px; border-bottom:1px solid #edf1f6; }
.score-pill { display:inline-block; padding:3px 7px; border-radius:999px; font-size:8px; font-weight:800; background:#edf4ff; color:#3267c7; }
.status-pill { display:inline-block; padding:3px 8px; border-radius:999px; font-size:8px; font-weight:800; background:#eef4ff; color:#3267c7; }
.bottom-card { min-height:112px; }
.skill-pill { display:inline-block; background:#f2f5fb; color:#50617f; border-radius:999px; padding:5px 8px; font-size:8px; margin:3px; }
.activity-mini { display:flex; gap:8px; padding:8px 0; border-bottom:1px solid #edf1f6; }
.activity-mini:last-child { border-bottom:0; }
.activity-mini-icon { width:25px; height:25px; border-radius:8px; background:#eef4ff; display:flex; align-items:center; justify-content:center; font-size:12px; }
.activity-mini-title { color:#26344d; font-size:9px; font-weight:700; }
.activity-mini-copy { color:#8a95a8; font-size:8px; margin-top:2px; }
.profile-top { display:flex; align-items:center; gap:9px; justify-content:flex-end; padding-top:2px; }
.profile-top-avatar { width:34px; height:34px; border-radius:50%; background:#e7edf8; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:800; color:#2d4770; }
.profile-top-name { color:#17213a; font-size:11px; font-weight:800; }
.profile-top-role { color:#7a86a0; font-size:9px; }
.seeker-label {
    margin-top: 10px !important;
}
.sidebar-divider {
    height: 1px;
    background: #2a3953;
    margin: 8px 0 8px;
}
.profile-copy {
    flex: 1;
}
.profile-chevron {
    color: #8fa2bf;
    font-size: 17px;
    padding-right: 4px;
}

/* ---------- Header ---------- */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 18px;
}
.top-title {
    color: #101828;
    font-size: 31px;
    font-weight: 780;
    letter-spacing: -.03em;
}
.top-subtitle {
    color: #64748b;
    margin-top: 3px;
    font-size: 14px;
}
.hero-banner {
    border-radius: 14px;
    padding: 17px 22px;
    background: linear-gradient(100deg, #f1f6ff, #e9e4ff);
    border: 1px solid #dfe5f4;
    min-height: 86px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.hero-robot {
    font-size: 38px;
}
.hero-title { font-weight: 750; color: #182238; font-size: 15px; }
.hero-copy { color: #64748b; font-size: 12px; margin-top: 4px; }

section[data-testid="stSidebar"] {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    height: 100vh !important;
}

section[data-testid="stSidebar"] > div:first-child {
    height: 100vh !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    height: 100vh !important;
    overflow: hidden !important;
}

section[data-testid="stSidebar"] .stVerticalBlock {
    gap: 0.25rem !important;
}

/* ---------- Cards ---------- */
.card {
    background: #fff;
    border: 1px solid #e6ebf3;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, .035);
}
.feature-card { min-height: 128px; }
.feature-icon {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    margin-bottom: 11px;
    background: #edf4ff;
}
.card-title { color: #172033; font-weight: 720; font-size: 14px; }
.card-copy { color: #667085; font-size: 12px; line-height: 1.45; margin-top: 4px; }
.card-arrow { color: #2f6fe5; float: right; font-size: 18px; }
.section-title { color: #172033; font-size: 17px; font-weight: 750; margin: 16px 0 10px; }

/* ---------- Metrics ---------- */
.metric-card {
    background: #fff;
    border: 1px solid #e6ebf3;
    border-radius: 12px;
    padding: 13px;
    min-height: 88px;
}
.metric-label { color: #667085; font-size: 11px; }
.metric-value { color: #172033; font-size: 25px; font-weight: 780; margin-top: 4px; }
.metric-trend { color: #16a34a; font-size: 10px; margin-top: 3px; }

/* ---------- Hiring agent ---------- */
.agent-shell {
    background: #fff;
    border: 1px solid #e2e8f2;
    border-radius: 15px;
    padding: 18px;
    box-shadow: 0 2px 10px rgba(15,23,42,.035);
}
.agent-heading { display: flex; align-items: center; gap: 11px; }
.agent-icon {
    width: 43px; height: 43px; border-radius: 12px;
    display:flex; align-items:center; justify-content:center;
    background: #eee8ff; font-size: 22px;
}
.agent-title { font-size: 18px; font-weight: 780; color: #172033; }
.agent-subtitle { font-size: 12px; color: #7a8599; }
.pill {
    display: inline-block; padding: 4px 9px; border-radius: 999px;
    background: #eef3ff; color: #2d63c8; font-size: 10px; font-weight: 700;
}
.stepper { display:flex; align-items:flex-start; justify-content:space-between; margin: 20px 5px 14px; }
.step { flex:1; text-align:center; position:relative; }
.step:not(:last-child):after {
    content:""; position:absolute; top:14px; left:58%; right:-42%; height:1px; background:#dce3ef;
}
.step-dot {
    width:29px; height:29px; border-radius:50%; margin:0 auto 7px;
    display:flex; align-items:center; justify-content:center;
    background:#3173e7; color:#fff; font-size:12px; font-weight:700; position:relative; z-index:1;
}
.step-name { font-size:11px; color:#26344d; font-weight:700; }
.step-copy { font-size:9px; color:#8a95a8; margin-top:2px; }
.info-box { border:1px solid #e7ebf2; border-radius:10px; padding:13px; background:#fbfcfe; min-height:126px; }
.info-box.green { background:#f1fbf5; border-color:#bfe8ce; }
.info-box-title { font-weight:700; font-size:12px; color:#24304a; }
.info-box-copy { color:#667085; font-size:11px; line-height:1.5; margin-top:7px; }

/* ---------- Activity ---------- */
.activity-item { display:flex; gap:10px; padding:10px 0; border-bottom:1px solid #edf0f5; }
.activity-icon { width:30px; height:30px; border-radius:9px; background:#eef4ff; display:flex; align-items:center; justify-content:center; }
.activity-title { color:#28344b; font-size:11px; font-weight:700; }
.activity-copy { color:#7b879a; font-size:10px; margin-top:2px; }
.activity-time { color:#98a2b3; font-size:9px; margin-left:auto; white-space:nowrap; }

/* buttons */
.stButton > button { border-radius: 9px; font-weight: 650; }
</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

PAGE_OPTIONS = [
    ("🏠", "Dashboard", "🏠 Dashboard"),
    ("💼", "Jobs", "💼 Job Management"),
    ("👥", "Candidates", "👥 Candidate Screening"),
    ("🤖", "AI Hiring Agent", "🤖 AI Hiring Agent"),
    ("📚", "RAG Assistant", "📚 RAG Knowledge"),
    ("🔄", "Pipeline", "📈 Career Dashboard"),
    ("📅", "Interviews", "🎙️ Interview Assistant"),
    ("📊", "Analytics", "📊 Analytics"),
    ("🔔", "Notifications", "📈 Career Dashboard"),
]

SEEKER_PAGES = [
    ("📄", "Resume Optimizer", "📄 Resume Optimizer"),
    ("📝", "Resume Builder", "📝 Resume Builder"),
    ("✉️", "Job Application Assistant", "✉️ Job Application Assistant"),
    ("🎯", "Career Guidance", "🎯 Career Guidance"),
]

if "active_page" not in st.session_state:
    st.session_state.active_page = "🏠 Dashboard"

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-row">
            <div class="brand-icon">🧠</div>
            <div>
              <div class="brand-name">AI Hiring Platform</div>
              <div class="brand-sub">Recruit Smart • Build Better</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nav-label">Recruiter Workspace</div>', unsafe_allow_html=True)

    for icon, label, page_value in PAGE_OPTIONS:
        is_active = st.session_state.active_page == page_value
        if st.button(
            f"{icon}  {label}",
            key=f"main_sidebar_{label}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.active_page = page_value
            st.rerun()

    st.markdown('<div class="nav-label seeker-label">For Job Seekers</div>', unsafe_allow_html=True)

    for icon, label, page_value in SEEKER_PAGES:
        is_active = st.session_state.active_page == page_value
        if st.button(
            f"{icon}  {label}",
            key=f"seeker_sidebar_{label}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.active_page = page_value
            st.rerun()

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sidebar-profile">
          <div class="avatar">{(get_user_name() or "U")[:2].upper()}</div>
          <div class="profile-copy">
            <div class="profile-name">{get_user_name() or "User"}</div>
            <div class="profile-role">{get_user_role() or "Account"}</div>
          </div>
          <div class="profile-chevron">⌄</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("↪ Logout", key="sidebar_logout", use_container_width=True):
        sign_out()
        st.rerun()

# ============================================================
# TOP SEARCH / HEADER
# ============================================================

header_col, search_col, alert_col = st.columns([2.2, 4.4, 0.8])
with search_col:
    search_query = st.text_input(
        "Search",
        placeholder="Search jobs, candidates, or features...",
        label_visibility="collapsed",
        key="global_search",
    )
with alert_col:
    st.markdown("<div style='text-align:center;font-size:24px;padding-top:4px'>🔔</div>", unsafe_allow_html=True)


# ============================================================
# DASHBOARD
# ============================================================

if "active_page" not in st.session_state:
    st.session_state.active_page = "🏠 Dashboard"

active_page = st.session_state.active_page
if active_page == "🏠 Dashboard":

    candidates = st.session_state.candidate_records
    total_jobs = len(st.session_state.saved_jobs)
    total_candidates = len(candidates)
    screened_candidates = sum(
        1 for c in candidates
        if c.get("screening_history") or c.get("ats_match_score", 0) or c.get("status") in {"Screening", "Shortlisted", "Interview", "Hired"}
    )
    interviews_scheduled = sum(
        bool(c.get("interview", {}).get("date")) or c.get("status") == "Interview"
        for c in candidates
    )
    hired_count = sum(c.get("status") == "Hired" for c in candidates)

    # Header/profile row
    header_left, header_search, header_profile = st.columns([1.25, 4.2, 1.35])
    with header_search:
        st.text_input(
            "Global Search",
            placeholder="Search candidates, jobs, or anything...",
            label_visibility="collapsed",
            key="global_search",
        )
    with header_profile:
        user_name = get_user_name() or "User"
        user_role = get_user_role() or "Recruiter"
        initials = "".join([part[0] for part in user_name.split()[:2]]).upper() or "U"
        st.markdown(
            f"""
            <div class="profile-top">
              <div style="font-size:18px">🔔</div>
              <div class="profile-top-avatar">{initials}</div>
              <div>
                <div class="profile-top-name">{user_name}</div>
                <div class="profile-top-role">{user_role}</div>
              </div>
              <div style="color:#60708d;font-size:13px">⌄</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Main dashboard grid: large welcome panel + quick actions.
    left_top, right_top = st.columns([3.15, 1.0])
    with left_top:
        st.markdown(
            f"""
            <div class="dashboard-hero">
              <div class="dashboard-robot">🤖</div>
              <div>
                <div class="dashboard-hero-title">👋 Good morning, {user_name}!</div>
                <div class="dashboard-hero-copy">Here's what's happening with your hiring today.</div>
              </div>
              <div class="dashboard-hero-date">September 27, 2026<br><span class="pill">Recruiter Dashboard</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right_top:
        st.markdown(
            """
            <div class="quick-card">
              <div class="quick-title">Quick Actions</div>
              <div class="quick-btn">▣ &nbsp; Post a New Job</div>
              <div class="quick-btn">♙ &nbsp; Screen Candidates</div>
              <div class="quick-btn">✦ &nbsp; AI Hiring Agent</div>
              <div class="quick-btn">▣ &nbsp; Ask RAG Assistant</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        qa1, qa2 = st.columns(2)
        with qa1:
            if st.button("Post Job", key="dash_quick_job", use_container_width=True):
                st.session_state.active_page = "💼 Job Management"
                st.rerun()
        with qa2:
            if st.button("Screen", key="dash_quick_screen", use_container_width=True):
                st.session_state.active_page = "👥 Candidate Screening"
                st.rerun()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Four KPI cards.
    metrics = [
        ("💼", "Active Jobs", total_jobs, "↑ 2"),
        ("▣", "Total Applications", total_candidates, "↑ 18%"),
        ("♙", "Screened Candidates", screened_candidates, "↑ 12%"),
        ("▣", "Interviews Scheduled", interviews_scheduled, "↑ 5%"),
    ]
    metric_cols = st.columns(4)
    for col, (icon, label, value, trend) in zip(metric_cols, metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-label">{icon} &nbsp; {label}</div>
                  <div class="metric-value">{value}</div>
                  <div class="metric-trend">{trend}</div>
                  <div class="dash-muted">vs last 7 days</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Main analytics area + right rail.
    left_col, rail_col = st.columns([2.75, 1.0])

    with left_col:
        funnel_col, trend_col = st.columns([1.15, 1.0])

        with funnel_col:
            st.markdown('<div class="dash-card"><div class="dash-card-title">Hiring Funnel <span class="dash-muted">Last 30 days ▾</span></div>', unsafe_allow_html=True)
            applied = max(total_candidates, 0)
            screening = screened_candidates
            shortlisted = sum(1 for c in candidates if c.get("status") == "Shortlisted")
            interview = interviews_scheduled
            offer = sum(1 for c in candidates if c.get("status") in {"Offer", "Offered"})
            hired = hired_count
            funnel = [
                ("Applied", applied, 100, 100),
                ("Screening", screening, 35, 86),
                ("Shortlisted", shortlisted, 17, 76),
                ("Interview", interview, 10, 65),
                ("Offer", offer, 3, 54),
                ("Hired", hired, 2, 45),
            ]
            widths = [96, 84, 72, 60, 50, 42]
            for (name, value, pct, _), width in zip(funnel, widths):
                st.markdown(
                    f"<div class='funnel-row' style='width:{width}%;background:#eef4ff'><span>{name}</span><span>{value} &nbsp;&nbsp; {pct}%</span></div>",
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        with trend_col:
            # Use workflow totals to keep the chart data tied to the current session.
            base = max(total_candidates, 1)
            series_a = [max(1, int(base * x)) for x in (0.48, 0.56, 0.50, 0.65, 0.61, 0.76, 0.70, 0.88, 0.84, 1.00)]
            series_b = [max(1, int(max(interviews_scheduled, 1) * x)) for x in (0.45, 0.38, 0.55, 0.48, 0.68, 0.58, 0.72, 0.62, 0.82, 0.76)]
            w, h = 430, 190
            def points(vals):
                vmax = max(max(vals), 1)
                return " ".join(f"{12 + i*(w-24)/(len(vals)-1):.1f},{h-18-(v/vmax)*(h-42):.1f}" for i,v in enumerate(vals))
            p1 = points(series_a)
            p2 = points(series_b)
            st.markdown(
                f"""
                <div class="dash-card trend-box">
                  <div class="dash-card-title">Applications Trend</div>
                  <div class="dash-muted">● Total Applications &nbsp;&nbsp; <span style='color:#6d35e8'>● Hired</span></div>
                  <svg class="trend-svg" viewBox="0 0 {w} {h}" preserveAspectRatio="none">
                    <line x1="12" y1="32" x2="{w-12}" y2="32" stroke="#edf1f7"/>
                    <line x1="12" y1="82" x2="{w-12}" y2="82" stroke="#edf1f7"/>
                    <line x1="12" y1="132" x2="{w-12}" y2="132" stroke="#edf1f7"/>
                    <polyline fill="none" stroke="#2f6fe5" stroke-width="2.5" points="{p1}"/>
                    <polyline fill="none" stroke="#6d35e8" stroke-width="2.5" points="{p2}"/>
                  </svg>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Recent candidates table.
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        rows = candidates[-5:][::-1]
        if rows:
            html_rows = []
            for c in rows:
                candidate_name = c.get("candidate_name") or "Candidate"
                job_title = c.get("job_title") or "—"
                company = c.get("company") or "—"
                ats = c.get("ats_match_score", 0) or 0
                agent = (c.get("hiring_agent_analysis") or {}).get("resume_alignment_score", "—")
                status = c.get("status") or "Applied"
                html_rows.append(
                    f"<tr><td><b>{candidate_name}</b><br><span class='dash-muted'>{c.get('file_name','')}</span></td><td>{job_title}<br><span class='dash-muted'>{company}</span></td><td><span class='score-pill'>{ats}%</span></td><td><span class='score-pill'>{agent}%</span></td><td><span class='status-pill'>{status}</span></td><td>{c.get('next_action_date') or '—'}</td><td>•••</td></tr>"
                )
        else:
            html_rows = ["<tr><td colspan='7' style='text-align:center;padding:20px;color:#8a95a8'>No candidates yet. Start by screening a resume.</td></tr>"]
        st.markdown(
            """
            <div class="dash-card">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <div class="dash-card-title">Recent Candidates</div><span class="dash-muted">View All →</span>
              </div>
              <table class="candidate-table">
                <thead><tr><th>Name</th><th>Job Title</th><th>ATS Score</th><th>AI Score</th><th>Status</th><th>Applied</th><th></th></tr></thead>
                <tbody>""" + "".join(html_rows) + """</tbody>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        bottom1, bottom2, bottom3 = st.columns(3)
        with bottom1:
            latest_job = st.session_state.saved_jobs[-1] if st.session_state.saved_jobs else None
            job_title = latest_job.get("job_title", "No jobs yet") if isinstance(latest_job, dict) else "No jobs yet"
            st.markdown(
                f"<div class='dash-card bottom-card'><div class='dash-card-title'>Recent Jobs</div><div style='margin-top:12px;font-size:11px;font-weight:700;color:#26344d'>💼 {job_title}</div><div class='dash-muted' style='margin-top:4px'>View All →</div></div>",
                unsafe_allow_html=True,
            )
        with bottom2:
            skill_counts = {}
            for c in candidates:
                for skill in c.get("matching_skills", []) or []:
                    key = str(skill)
                    skill_counts[key] = skill_counts.get(key, 0) + 1
            skills = sorted(skill_counts, key=skill_counts.get, reverse=True)[:5] or ["Python", "React", "Node.js", "SQL", "APIs"]
            pills = "".join(f"<span class='skill-pill'>{s}</span>" for s in skills)
            st.markdown(
                f"<div class='dash-card bottom-card'><div class='dash-card-title'>Top Skills <span class='dash-muted'>(From Recent Applications)</span></div><div style='margin-top:10px'>{pills}</div></div>",
                unsafe_allow_html=True,
            )
        with bottom3:
            st.markdown(
                "<div class='dash-card bottom-card'><div class='dash-card-title'>💡 Hiring Insights</div><div class='dash-muted' style='margin-top:12px'>Candidates with strong skill alignment and completed screening move faster through the workflow.</div><div style='margin-top:9px;color:#3267c7;font-size:9px;font-weight:700'>View Details →</div></div>",
                unsafe_allow_html=True,
            )

    with rail_col:
        # AI Hiring Agent rail card
        st.markdown(
            """
            <div class="rail-card" style="padding:14px;margin-bottom:12px">
              <div class="dash-card-title">✦ AI Hiring Agent <span style="float:right;color:#3267c7;font-size:9px">View All →</span></div>
              <div class="dash-muted" style="margin-top:8px;line-height:1.5">Let AI analyze candidates, match skills and provide hiring recommendations.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Run AI Hiring Agent", key="dash_rail_agent", use_container_width=True, type="primary"):
            st.session_state.active_page = "🤖 AI Hiring Agent"
            st.rerun()

        # RAG rail card
        st.markdown(
            """
            <div class="rail-card" style="padding:14px;margin:12px 0">
              <div class="dash-card-title">📖 RAG Knowledge Assistant <span style="float:right;color:#3267c7;font-size:9px">View All →</span></div>
              <div class="dash-muted" style="margin-top:8px;line-height:1.5">Ask questions about your company handbook, policies and documents.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        rag_question = st.text_input("RAG question", placeholder="Ask anything...", label_visibility="collapsed", key="dashboard_rag_question")
        if st.button("→ Ask RAG Assistant", key="dash_rail_rag", use_container_width=True):
            if rag_question.strip():
                st.session_state.rag_answer = None
                st.session_state.active_page = "📚 RAG Knowledge"
                st.rerun()
        st.markdown("<div class='dash-muted' style='margin:8px 0 5px'>Popular questions:</div><div class='skill-pill'>What is our interview process?</div><div class='skill-pill'>Company leave policy</div><div class='skill-pill'>Technical interview guidelines</div>", unsafe_allow_html=True)

        # Activity rail
        activity = []
        if st.session_state.hiring_agent_result:
            r = st.session_state.hiring_agent_result
            activity.append(("♙", "New candidate analyzed", r.get("candidate_name", "Candidate"), "recently"))
        if candidates:
            activity.append(("▣", "Candidate moved to workflow", candidates[-1].get("candidate_name", "Candidate"), "recently"))
        if interviews_scheduled:
            activity.append(("♧", "Interview scheduled", f"{interviews_scheduled} candidate(s)", "recently"))
        if st.session_state.rag_knowledge:
            activity.append(("▣", "New message in RAG assistant", "Knowledge base available", "1h ago"))
        if not activity:
            activity = [("✦", "Workspace ready", "Start by creating a job", "now")]
        activity_html = "".join(
            f"<div class='activity-mini'><div class='activity-mini-icon'>{icon}</div><div><div class='activity-mini-title'>{title}</div><div class='activity-mini-copy'>{copy} · {when}</div></div></div>"
            for icon,title,copy,when in activity[:5]
        )
        st.markdown(
            f"<div class='rail-card' style='padding:14px;margin-top:12px'><div class='dash-card-title'>♧ Recent Activity <span style='float:right;color:#3267c7;font-size:9px'>View All →</span></div>{activity_html}</div>",
            unsafe_allow_html=True,
        )


# ============================================================
# JOB SEEKER
# ============================================================

# ============================================================

if active_page in ["👤 Job Seeker", "📄 Resume Optimizer", "📝 Resume Builder", "✉️ Job Application Assistant", "🎯 Career Guidance"]:

    st.subheader(
        "👤 Job Seeker Tools"
    )

    sidebar_tool_map = {
        "📄 Resume Optimizer": "Resume Optimizer",
        "📝 Resume Builder": "Resume Builder",
        "✉️ Job Application Assistant": "Job Application Assistant",
        "🎯 Career Guidance": "Interview Assistant",
    }
    default_tool = sidebar_tool_map.get(active_page, "Resume Optimizer")
    tool = st.selectbox(
        "Select a tool",
        [
            "Resume Optimizer",
            "Resume Builder",
            "Job Application Assistant",
            "Interview Assistant",
        ],
        index=["Resume Optimizer", "Resume Builder", "Job Application Assistant", "Interview Assistant"].index(default_tool),
        key="job_seeker_tool_selector"
    )

    # ========================================================
    # RESUME OPTIMIZER
    # ========================================================

    if tool == "Resume Optimizer":

        st.header(
            "📄 Resume Optimizer"
        )

        resume_file = st.file_uploader(
            "Upload Resume PDF",
            type=["pdf"],
            key="job_seeker_optimizer_resume"
        )

        job_description = st.text_area(
            "Target Job Description",
            height=250,
            key="optimizer_jd"
        )

        if st.button(
            "🚀 Optimize Resume",
            key="job_seeker_optimize_button"
        ):

            if not resume_file:

                st.warning(
                    "Please upload a resume PDF."
                )

            elif not job_description.strip():

                st.warning(
                    "Please provide a job description."
                )

            else:

                with st.spinner(
                    "Analyzing resume..."
                ):

                    try:

                        result = optimize_resume(
                            client,
                            resume_file,
                            job_description
                        )

                        st.session_state.optimizer_result = result

                    except Exception as error:

                        st.error(
                            format_gemini_error(
                                error
                            )
                        )

        result = st.session_state.optimizer_result

        if result:

            st.divider()

            score = result.get(
                "optimization_score",
                0
            )

            st.metric(
                "Optimization Score",
                f"{score}/100"
            )

            st.progress(
                max(
                    0,
                    min(
                        100,
                        int(score)
                    )
                ) / 100
            )

            st.subheader(
                "Resume Strengths"
            )

            for item in safe_list(
                result.get(
                    "resume_strengths"
                )
            ):

                st.success(
                    str(item)
                )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "Keywords to Emphasize"
                )

                for item in safe_list(
                    result.get(
                        "keywords_to_emphasize"
                    )
                ):

                    st.write(
                        f"• {item}"
                    )

            with col2:

                st.subheader(
                    "Missing Job Keywords"
                )

                for item in safe_list(
                    result.get(
                        "missing_job_keywords"
                    )
                ):

                    st.write(
                        f"• {item}"
                    )

            alignment = result.get(
                "skills_alignment",
                {}
            )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "Matching Skills"
                )

                for item in safe_list(
                    alignment.get(
                        "matching_skills"
                    )
                ):

                    st.write(
                        f"✓ {item}"
                    )

            with col2:

                st.subheader(
                    "Missing Skills"
                )

                for item in safe_list(
                    alignment.get(
                        "missing_skills"
                    )
                ):

                    st.write(
                        f"• {item}"
                    )

            summary = result.get(
                "professional_summary",
                {}
            )

            st.subheader(
                "Professional Summary Assessment"
            )

            st.write(
                summary.get(
                    "current_assessment",
                    ""
                )
            )

            st.subheader(
                "Suggested Summary"
            )

            st.info(
                summary.get(
                    "suggestion",
                    ""
                )
            )

            st.subheader(
                "Experience Improvements"
            )

            for item in safe_list(
                result.get(
                    "experience_improvements"
                )
            ):

                with st.expander(
                    item.get(
                        "original",
                        "Experience"
                    )
                ):

                    st.write(
                        "**Suggested:** "
                        + item.get(
                            "suggested",
                            ""
                        )
                    )

                    st.write(
                        "**Reason:** "
                        + item.get(
                            "reason",
                            ""
                        )
                    )

            st.subheader(
                "ATS Formatting Suggestions"
            )

            for item in safe_list(
                result.get(
                    "ats_formatting_suggestions"
                )
            ):

                st.write(
                    f"• {item}"
                )

            st.subheader(
                "Overall Suggestions"
            )

            for item in safe_list(
                result.get(
                    "overall_suggestions"
                )
            ):

                st.write(
                    f"• {item}"
                )

            warning = result.get(
                "important_warning",
                ""
            )

            if warning:

                st.warning(
                    warning
                )


    # ========================================================
    # RESUME BUILDER
    # ========================================================

    elif tool == "Resume Builder":

        st.header(
            "📝 Resume Builder"
        )

        resume_file = st.file_uploader(
            "Upload Current Resume PDF",
            type=["pdf"],
            key="builder_resume"
        )

        job_description = st.text_area(
            "Target Job Description",
            height=250,
            key="builder_jd"
        )

        if st.button(
            "✨ Build Resume",
            key="builder_button"
        ):

            if not resume_file:

                st.warning(
                    "Please upload a resume PDF."
                )

            elif not job_description.strip():

                st.warning(
                    "Please provide a job description."
                )

            else:

                with st.spinner(
                    "Building resume..."
                ):

                    try:

                        result = build_resume(
                            client,
                            resume_file,
                            job_description
                        )

                        st.session_state.builder_result = result

                    except Exception as error:

                        st.error(
                            format_gemini_error(
                                error
                            )
                        )

        resume = st.session_state.builder_result

        if resume:

            st.divider()

            st.subheader(
                resume.get(
                    "candidate_name",
                    "Candidate"
                )
            )

            st.write(
                resume.get(
                    "professional_title",
                    ""
                )
            )

            contact = resume.get(
                "contact_information",
                {}
            )

            if contact:

                st.write(
                    " | ".join(
                        str(value)
                        for value in contact.values()
                        if value
                    )
                )

            st.subheader(
                "Professional Summary"
            )

            st.write(
                resume.get(
                    "professional_summary",
                    ""
                )
            )

            st.subheader(
                "Skills"
            )

            st.write(
                ", ".join(
                    map(
                        str,
                        safe_list(
                            resume.get(
                                "skills"
                            )
                        )
                    )
                )
            )

            st.subheader(
                "Work Experience"
            )

            for job in safe_list(
                resume.get(
                    "work_experience"
                )
            ):

                st.markdown(
                    f"**{job.get('job_title', '')}** "
                    f"— {job.get('company', '')}"
                )

                for item in safe_list(
                    job.get(
                        "responsibilities"
                    )
                ):

                    st.write(
                        f"• {item}"
                    )

            st.subheader(
                "Projects"
            )

            for project in safe_list(
                resume.get(
                    "projects"
                )
            ):

                st.markdown(
                    f"**{project.get('project_name', '')}**"
                )

                st.write(
                    project.get(
                        "description",
                        ""
                    )
                )

            st.subheader(
                "Education"
            )

            for item in safe_list(
                resume.get(
                    "education"
                )
            ):

                st.write(
                    f"**{item.get('degree', '')}** — "
                    f"{item.get('institution', '')} "
                    f"{item.get('dates', '')}"
                )

            pdf_bytes = create_resume_pdf(
                resume
            )

            st.download_button(
                "📥 Download Professional Resume",
                data=pdf_bytes,
                file_name="professional_resume.pdf",
                mime="application/pdf"
            )

            st.warning(
                resume.get(
                    "truthfulness_warning",
                    "Verify all information before using the generated resume."
                )
            )


    # ========================================================
    # APPLICATION ASSISTANT
    # ========================================================

    elif tool == "Job Application Assistant":

        st.header(
            "📨 Job Application Assistant"
        )

        resume_file = st.file_uploader(
            "Upload Resume PDF",
            type=["pdf"],
            key="application_resume"
        )

        job_description = st.text_area(
            "Target Job Description",
            height=250,
            key="application_jd"
        )

        if st.button(
            "🚀 Generate Application Package",
            key="application_button"
        ):

            if not resume_file:

                st.warning(
                    "Please upload a resume PDF."
                )

            elif not job_description.strip():

                st.warning(
                    "Please provide a job description."
                )

            else:

                with st.spinner(
                    "Generating application package..."
                ):

                    try:

                        result = generate_application_package(
                            client,
                            resume_file,
                            job_description
                        )

                        st.session_state.application_result = result

                    except Exception as error:

                        st.error(
                            format_gemini_error(
                                error
                            )
                        )

        result = st.session_state.application_result

        if result:

            package = result.get(
                "application_package",
                {}
            )

            alignment = result.get(
                "job_alignment",
                {}
            )

            st.divider()

            st.subheader(
                "Cover Letter"
            )

            st.text_area(
                "Cover Letter",
                package.get(
                    "cover_letter",
                    ""
                ),
                height=300
            )

            email = package.get(
                "application_email",
                {}
            )

            st.subheader(
                "Application Email"
            )

            st.text_input(
                "Subject",
                email.get(
                    "subject",
                    ""
                )
            )

            st.text_area(
                "Email Body",
                email.get(
                    "body",
                    ""
                ),
                height=250
            )

            st.subheader(
                "Recruiter / LinkedIn Message"
            )

            st.text_area(
                "Message",
                package.get(
                    "recruiter_message",
                    ""
                ),
                height=180
            )

            st.subheader(
                "Key Talking Points"
            )

            for item in safe_list(
                package.get(
                    "key_talking_points"
                )
            ):

                st.write(
                    f"• {item}"
                )

            st.subheader(
                "Matching Qualifications"
            )

            for item in safe_list(
                alignment.get(
                    "matching_qualifications"
                )
            ):

                st.success(
                    str(item)
                )

            st.subheader(
                "Missing Requirements"
            )

            for item in safe_list(
                alignment.get(
                    "missing_requirements"
                )
            ):

                st.write(
                    f"• {item}"
                )

            warning = result.get(
                "truthfulness_warning",
                ""
            )

            if warning:

                st.warning(
                    warning
                )


    # ========================================================
    # INTERVIEW ASSISTANT
    # ========================================================

    elif tool == "Interview Assistant":

        st.header(
            "🎯 Interview Assistant"
        )

        resume_file = st.file_uploader(
            "Upload Resume PDF",
            type=["pdf"],
            key="interview_resume"
        )

        job_description = st.text_area(
            "Target Job Description",
            height=250,
            key="interview_jd"
        )

        if st.button(
            "🎯 Generate Interview Preparation",
            key="interview_button"
        ):

            if not resume_file:

                st.warning(
                    "Please upload a resume PDF."
                )

            elif not job_description.strip():

                st.warning(
                    "Please provide a job description."
                )

            else:

                with st.spinner(
                    "Preparing interview..."
                ):

                    try:

                        result = generate_interview_preparation(
                            client,
                            resume_file,
                            job_description
                        )

                        st.session_state.interview_result = result

                    except Exception as error:

                        st.error(
                            format_gemini_error(
                                error
                            )
                        )

        result = st.session_state.interview_result

        if result:

            st.divider()

            overview = result.get(
                "interview_overview",
                {}
            )

            st.subheader(
                "Interview Overview"
            )

            st.write(
                overview.get(
                    "role_focus",
                    ""
                )
            )

            st.write(
                overview.get(
                    "preparation_summary",
                    ""
                )
            )

            st.subheader(
                "Key Topics"
            )

            for item in safe_list(
                overview.get(
                    "key_topics"
                )
            ):

                st.write(
                    f"• {item}"
                )

            st.subheader(
                "Technical Questions"
            )

            for index, item in enumerate(
                safe_list(
                    result.get(
                        "technical_questions"
                    )
                ),
                1
            ):

                with st.expander(
                    f"{index}. {item.get('question', '')}"
                ):

                    st.write(
                        "**Why it may be asked:**"
                    )

                    st.write(
                        item.get(
                            "why_it_may_be_asked",
                            ""
                        )
                    )

                    st.write(
                        "**Answer guidance:**"
                    )

                    st.write(
                        item.get(
                            "answer_guidance",
                            ""
                        )
                    )

                    st.write(
                        "**Resume support:**"
                    )

                    st.write(
                        item.get(
                            "resume_support",
                            ""
                        )
                    )

                    st.write(
                        "**Difficulty:** "
                        + item.get(
                            "difficulty",
                            ""
                        )
                    )

            st.subheader(
                "Behavioral Questions"
            )

            for item in safe_list(
                result.get(
                    "behavioral_questions"
                )
            ):

                with st.expander(
                    item.get(
                        "question",
                        ""
                    )
                ):

                    st.write(
                        item.get(
                            "answer_guidance",
                            ""
                        )
                    )

            st.subheader(
                "Resume-Based Questions"
            )

            for item in safe_list(
                result.get(
                    "resume_based_questions"
                )
            ):

                with st.expander(
                    item.get(
                        "question",
                        ""
                    )
                ):

                    st.write(
                        "**Resume Reference:** "
                        + item.get(
                            "resume_reference",
                            ""
                        )
                    )

                    st.write(
                        "**Answer Guidance:** "
                        + item.get(
                            "answer_guidance",
                            ""
                        )
                    )

            st.subheader(
                "Job-Specific Questions"
            )

            for item in safe_list(
                result.get(
                    "job_specific_questions"
                )
            ):

                with st.expander(
                    item.get(
                        "question",
                        ""
                    )
                ):

                    st.write(
                        "**Requirement:** "
                        + item.get(
                            "job_requirement",
                            ""
                        )
                    )

                    st.write(
                        "**Answer Guidance:** "
                        + item.get(
                            "answer_guidance",
                            ""
                        )
                    )

            st.subheader(
                "Potential Weak Areas"
            )

            for item in safe_list(
                result.get(
                    "potential_weak_areas"
                )
            ):

                st.warning(
                    f"**{item.get('area', '')}**\n\n"
                    f"{item.get('reason', '')}\n\n"
                    f"Preparation tip: "
                    f"{item.get('preparation_tip', '')}"
                )

            st.subheader(
                "Preparation Checklist"
            )

            for item in safe_list(
                result.get(
                    "preparation_checklist"
                )
            ):

                st.checkbox(
                    str(item),
                    key=f"interview_prep_{index}"
                )


# ============================================================
# RECRUITER MODE
# ============================================================

elif active_page in ["🏢 Recruiter Mode", "💼 Job Management", "👥 Candidate Screening", "🤖 AI Hiring Agent", "📚 RAG Knowledge", "📊 Analytics", "📈 Career Dashboard"]:

    st.subheader(
        "🏢 Recruiter Mode"
    )

    recruiter_module_map = {
        "💼 Job Management": "📋 Job Management",
        "👥 Candidate Screening": "🔍 Candidate Screening",
        "🤖 AI Hiring Agent": "🤖 AI Hiring Agent",
        "📚 RAG Knowledge": "🧠 RAG Knowledge Assistant",
        "📊 Analytics": "📈 Hiring Funnel Analytics",
        "📈 Career Dashboard": "📊 Recruiter Dashboard",
    }
    default_recruiter_module = recruiter_module_map.get(active_page, "📋 Job Management")
    recruiter_options = [
        "📋 Job Management",
        "🔍 Candidate Screening",
        "🤖 AI Hiring Agent",
        "📊 Recruiter Dashboard",
        "🔄 Candidate Pipeline",
        "👤 Candidate Workflow",
        "📧 Follow-up Assistant",
        "🧠 RAG Knowledge Assistant",
        "📈 Hiring Funnel Analytics",
        "📥 Export Workflow Data",
    ]
    recruiter_module = st.selectbox(
        "Select Recruiter Module",
        recruiter_options,
        index=recruiter_options.index(default_recruiter_module),
    )


    # ========================================================
    # JOB MANAGEMENT
    # ========================================================

    if recruiter_module == "📋 Job Management":

        st.header(
            "📋 Job Management"
        )

        job_title = st.text_input(
            "Job Title"
        )

        company = st.text_input(
            "Company"
        )

        job_description = st.text_area(
            "Job Description",
            height=250
        )

        required_skills = st.text_input(
            "Required Skills",
            placeholder="Python, SQL, APIs, AI..."
        )

        if st.button(
            "💾 Save Job"
        ):

            if not job_title.strip():

                st.warning(
                    "Enter a job title."
                )

            elif not job_description.strip():

                st.warning(
                    "Enter a job description."
                )

            else:

                job = {

                    "job_title":
                        job_title.strip(),

                    "company":
                        company.strip(),

                    "job_description":
                        job_description.strip(),

                    "required_skills":
                        required_skills.strip(),

                    "created_at":
                        get_now_string(),
                }

                st.session_state.saved_jobs.append(
                    job
                )

                st.success(
                    "Job saved successfully."
                )

        st.divider()

        st.subheader(
            "Saved Jobs"
        )

        if not st.session_state.saved_jobs:

            st.info(
                "No saved jobs yet."
            )

        else:

            for index, job in enumerate(
                st.session_state.saved_jobs
            ):

                with st.expander(
                    f"{job.get('job_title', '')} "
                    f"— {job.get('company', '')}"
                ):

                    st.write(
                        job.get(
                            "job_description",
                            ""
                        )
                    )

                    st.write(
                        "**Required Skills:** "
                        + job.get(
                            "required_skills",
                            ""
                        )
                    )

                    c1, c2 = st.columns(2)

                    with c1:

                        if st.button(
                            "Use This Job",
                            key=f"use_job_{index}"
                        ):

                            st.session_state.selected_job_index = index

                            st.success(
                                "Job selected."
                            )

                    with c2:

                        if st.button(
                            "Delete Job",
                            key=f"delete_job_{index}"
                        ):

                            st.session_state.saved_jobs.pop(
                                index
                            )

                            if (
                                st.session_state.selected_job_index
                                == index
                            ):

                                st.session_state.selected_job_index = None

                            st.rerun()

        selected_index = (
            st.session_state.selected_job_index
        )

        if (
            selected_index is not None
            and selected_index < len(
                st.session_state.saved_jobs
            )
        ):

            active_job = st.session_state.saved_jobs[
                selected_index
            ]

            st.success(
                f"Active Job: "
                f"{active_job.get('job_title', '')} "
                f"— "
                f"{active_job.get('company', '')}"
            )

            if st.button(
                "Clear Active Job"
            ):

                st.session_state.selected_job_index = None

                st.rerun()


    # ========================================================
    # CANDIDATE SCREENING
    # ========================================================

    elif recruiter_module == "🔍 Candidate Screening":

        st.header(
            "🔍 AI Candidate Screening"
        )

        active_job = None

        selected_index = (
            st.session_state.selected_job_index
        )

        if (
            selected_index is not None
            and selected_index < len(
                st.session_state.saved_jobs
            )
        ):

            active_job = st.session_state.saved_jobs[
                selected_index
            ]

            st.success(
                f"Using active job: "
                f"{active_job.get('job_title', '')}"
            )

            screening_job_title = active_job.get(
                "job_title",
                ""
            )

            screening_company = active_job.get(
                "company",
                ""
            )

            screening_jd = active_job.get(
                "job_description",
                ""
            )

        else:

            screening_job_title = st.text_input(
                "Job Title",
                key="screening_job_title"
            )

            screening_company = st.text_input(
                "Company",
                key="screening_company"
            )

            screening_jd = st.text_area(
                "Job Description",
                height=250,
                key="screening_jd"
            )

        resumes = st.file_uploader(
            "Upload Candidate Resumes",
            type=["pdf"],
            accept_multiple_files=True,
            key="candidate_resumes"
        )

        if st.button(
            "🔍 Screen Candidates"
        ):

            if not resumes:

                st.warning(
                    "Please upload at least one resume."
                )

            elif not screening_jd.strip():

                st.warning(
                    "Please provide a job description."
                )

            else:

                results = []

                progress = st.progress(
                    0
                )

                new_count = 0
                updated_count = 0

                for index, resume in enumerate(
                    resumes
                ):

                    try:

                        with st.spinner(
                            f"Analyzing {resume.name}..."
                        ):

                            result = screen_candidate(
                                client,
                                resume,
                                screening_jd
                            )

                        candidate, was_duplicate = (
                            create_candidate_from_screening(
                                result,
                                resume,
                                screening_job_title,
                                screening_company
                            )
                        )

                        if was_duplicate:

                            updated_count += 1

                        else:

                            new_count += 1

                        results.append({

                            "file_name":
                                resume.name,

                            "result":
                                result,

                            "candidate_id":
                                candidate.get(
                                    "candidate_id",
                                    ""
                                ),

                            "was_duplicate":
                                was_duplicate,
                        })

                    except Exception as error:

                        st.error(
                            f"{resume.name}: "
                            f"{format_gemini_error(error)}"
                        )

                    progress.progress(
                        (index + 1) / len(resumes)
                    )

                st.session_state.recruiter_results = results

                st.success(
                    "Screening completed."
                )

                if updated_count > 0:

                    st.info(
                        f"{updated_count} existing candidate "
                        f"record(s) were updated instead of "
                        f"creating duplicates."
                    )

                if new_count > 0:

                    st.info(
                        f"{new_count} new candidate "
                        f"record(s) added."
                    )

        if st.session_state.recruiter_results:

            st.divider()

            st.subheader(
                "Latest Screening Results"
            )

            for item in st.session_state.recruiter_results:

                result = item["result"]

                with st.expander(
                    result.get(
                        "candidate_name",
                        item["file_name"]
                    )
                ):

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "ATS Match",
                        f"{result.get('ats_match_score', 0)}/100"
                    )

                    c2.metric(
                        "Match Level",
                        result.get(
                            "match_level",
                            ""
                        )
                    )

                    c3.write(
                        f"**Resume:** "
                        f"{item['file_name']}"
                    )

                    if item.get(
                        "was_duplicate",
                        False
                    ):

                        st.caption(
                            "Existing candidate record updated."
                        )

                    else:

                        st.caption(
                            "New candidate record created."
                        )

                    st.subheader(
                        "Strengths"
                    )
                    for value in safe_list(
                        result.get(
                            "strengths"
                        )
                    ):
                        st.success(
                            str(value)
                        )

                    st.subheader(
                        "Matching Skills"
                    )

                    st.write(
                        ", ".join(
                            map(
                                str,
                                safe_list(
                                    result.get(
                                        "matching_skills"
                                    )
                                )
                            )
                        )
                    )
                    st.subheader(
                        "Missing Skills"
                    )

                    for value in safe_list(
                        result.get(
                            "missing_skills"
                        )
                    ):

                        st.write(
                            f"• {value}"
                        )

                    st.subheader(
                        "Gaps"
                    )

                    for value in safe_list(
                        result.get(
                            "gaps"
                        )
                    ):

                        st.write(
                            f"• {value}"
                        )

                    st.subheader(
                        "Recruiter Summary"
                    )

                    st.info(
                        result.get(
                            "recruiter_summary",
                            ""
                        )
                    )

            st.info(
                "AI screening provides structured decision-support "
                "information. Human review remains required."
            )


    # ========================================================
    # AI HIRING AGENT
    # ========================================================

    elif recruiter_module == "🤖 AI Hiring Agent":

        st.header(
            "🤖 AI Hiring Agent"
        )

        st.caption(
            "Recruitment Intelligence & Decision Support"
        )

        st.info(
            "The AI Hiring Agent analyzes a candidate resume "
            "against a job description and optionally uses "
            "company recruitment knowledge from the RAG "
            "Knowledge Assistant."
        )

        # ----------------------------------------------------
        # STEP 1 — JOB DESCRIPTION
        # ----------------------------------------------------

        st.subheader(
            "1️⃣ Job Description"
        )

        selected_index = (
            st.session_state.selected_job_index
        )

        active_job = None

        if (
            selected_index is not None
            and selected_index < len(
                st.session_state.saved_jobs
            )
        ):

            active_job = st.session_state.saved_jobs[
                selected_index
            ]

            st.success(
                f"Active Job: "
                f"{active_job.get('job_title', '')}"
                f" — "
                f"{active_job.get('company', '')}"
            )

            agent_job_title = active_job.get(
                "job_title",
                ""
            )

            agent_company = active_job.get(
                "company",
                ""
            )

            agent_job_description = active_job.get(
                "job_description",
                ""
            )

            st.text_area(
                "Active Job Description",
                value=agent_job_description,
                height=220,
                disabled=True,
                key="agent_active_jd"
            )

            use_active_job = True

        else:

            use_active_job = False

            agent_job_title = st.text_input(
                "Job Title",
                key="agent_job_title"
            )

            agent_company = st.text_input(
                "Company",
                key="agent_company"
            )

            agent_job_description = st.text_area(
                "Job Description",
                height=220,
                key="agent_jd"
            )

        st.divider()

        # ----------------------------------------------------
        # STEP 2 — CANDIDATE RESUME
        # ----------------------------------------------------

        st.subheader(
            "2️⃣ Candidate Resume"
        )

        agent_resume = st.file_uploader(
            "Upload Candidate Resume PDF",
            type=["pdf"],
            key="hiring_agent_resume"
        )

        if agent_resume:

            st.success(
                f"Resume uploaded: {agent_resume.name}"
            )

        st.divider()

        # ----------------------------------------------------
        # STEP 3 — RAG KNOWLEDGE
        # ----------------------------------------------------

        st.subheader(
            "3️⃣ RAG Knowledge"
        )

        agent_knowledge = (
            st.session_state.rag_knowledge
        )

        if agent_knowledge:

            document_count = agent_knowledge.get(
                "document_count",
                0
            )

            st.success(
                f"Company recruitment knowledge available "
                f"from {document_count} document(s)."
            )

            st.caption(
                "The AI Hiring Agent will use the currently "
                "loaded RAG Knowledge Base when relevant."
            )

            with st.expander(
                "View RAG Knowledge"
            ):

                for document in agent_knowledge.get(
                    "documents",
                    []
                ):

                    doc_name = document.get(
                        "document_name",
                        "Document"
                    )

                    doc_knowledge = document.get(
                        "knowledge",
                        {}
                    )

                    st.markdown(
                        f"### {doc_name}"
                    )

                    st.write(
                        "**Type:** "
                        + doc_knowledge.get(
                            "document_type",
                            ""
                        )
                    )

                    st.write(
                        "**Title:** "
                        + doc_knowledge.get(
                            "document_title",
                            ""
                        )
                    )

                    st.write(
                        "**Summary:** "
                        + doc_knowledge.get(
                            "summary",
                            ""
                        )
                    )

        else:

            st.warning(
                "No RAG Knowledge Base is currently loaded."
            )

            st.caption(
                "You can still run the AI Hiring Agent "
                "without company knowledge."
            )

        st.divider()

        # ----------------------------------------------------
        # STEP 4 — RUN AGENT
        # ----------------------------------------------------

        st.subheader(
            "4️⃣ Run AI Hiring Agent"
        )

        st.markdown(
            """
<div class="agent-card">

<b>AI Hiring Agent Analysis</b>

<p>
The agent will analyze the documented resume against
the provided job description and return structured
recruitment intelligence.
</p>

<p>
It will not make a hiring decision.
</p>

</div>
""",
            unsafe_allow_html=True
        )

        if st.button(
            "🚀 Run AI Hiring Agent",
            key="run_hiring_agent_button",
            use_container_width=True
        ):

            if not agent_resume:

                st.warning(
                    "Please upload a candidate resume PDF."
                )

            elif not agent_job_description.strip():

                st.warning(
                    "Please provide a job description."
                )

            else:

                with st.spinner(
                    "AI Hiring Agent is analyzing the candidate..."
                ):

                    try:

                        result = run_hiring_agent(
                            client,
                            agent_resume,
                            agent_job_description,
                            knowledge=agent_knowledge
                        )

                        st.session_state.hiring_agent_result = result

                        st.session_state.hiring_agent_saved_message = None

                        st.success(
                            "AI Hiring Agent analysis completed."
                        )

                    except Exception as error:

                        st.error(
                            format_gemini_error(
                                error
                            )
                        )

        # ----------------------------------------------------
        # STEP 5 — RESULTS
        # ----------------------------------------------------

        result = (
            st.session_state.hiring_agent_result
        )

        if result:

            st.divider()

            st.subheader(
                "5️⃣ AI Hiring Agent Results"
            )

            candidate_name = result.get(
                "candidate_name",
                "Unknown Candidate"
            )

            st.markdown(
                f"## 👤 {candidate_name}"
            )

            professional_profile = result.get(
                "professional_profile",
                ""
            )

            if professional_profile:

                st.info(
                    professional_profile
                )

            # ------------------------------------------------
            # SAVE TO CANDIDATE WORKFLOW
            # ------------------------------------------------

            st.markdown(
                """
<div class="agent-analysis-card">

<b>🔗 Candidate Workflow Integration</b>

<p>
Save this AI Hiring Agent analysis to the candidate's
workflow record. If the candidate already exists, the
existing record will be updated instead of creating a
duplicate.
</p>

<p>
The AI Agent alignment score will remain separate from
the normal ATS screening score.
</p>

</div>
""",
                unsafe_allow_html=True
            )

            if st.button(
                "💾 Save Analysis to Candidate Workflow",
                key="save_hiring_agent_analysis",
                use_container_width=True
            ):

                if not agent_resume:

                    st.warning(
                        "The original candidate resume is required "
                        "to save this analysis."
                    )

                elif not agent_job_description.strip():

                    st.warning(
                        "A job description is required."
                    )

                else:

                    try:

                        saved_candidate, was_duplicate = (
                            save_hiring_agent_to_candidate(
                                result,
                                agent_resume,
                                agent_job_title,
                                agent_company
                            )
                        )

                        if was_duplicate:

                            st.session_state.hiring_agent_saved_message = (
                                f"Analysis updated for "
                                f"{saved_candidate.get('candidate_name', 'candidate')} "
                                f"({saved_candidate.get('candidate_id', '')}). "
                                f"No duplicate candidate was created."
                            )

                        else:

                            st.session_state.hiring_agent_saved_message = (
                                f"Candidate "
                                f"{saved_candidate.get('candidate_name', 'candidate')} "
                                f"was added to Candidate Workflow with "
                                f"ID {saved_candidate.get('candidate_id', '')}."
                            )

                    except Exception as error:

                        st.error(
                            f"Unable to save AI Hiring Agent analysis: "
                            f"{error}"
                        )

            if st.session_state.hiring_agent_saved_message:

                st.success(
                    st.session_state.hiring_agent_saved_message
                )

            # ------------------------------------------------
            # ALIGNMENT SCORE
            # ------------------------------------------------

            st.subheader(
                "📊 Resume Alignment"
            )

            alignment_score = result.get(
                "resume_alignment_score",
                0
            )

            try:

                alignment_score = int(
                    alignment_score
                )

            except Exception:

                alignment_score = 0

            alignment_score = max(
                0,
                min(
                    100,
                    alignment_score
                )
            )

            score_col1, score_col2 = st.columns(
                [1, 2]
            )

            with score_col1:

                st.metric(
                    "Resume Alignment Score",
                    f"{alignment_score}/100"
                )

            with score_col2:

                st.progress(
                    alignment_score / 100
                )

                st.caption(
                    "This score measures documented resume "
                    "alignment with the job description. "
                    "It is not a hiring or candidate-quality score."
                )

            # ------------------------------------------------
            # MATCHING SKILLS / KEYWORDS
            # ------------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "✅ Matching Skills"
                )

                matching_skills = safe_list(
                    result.get(
                        "matching_skills"
                    )
                )

                if matching_skills:

                    for item in matching_skills:

                        st.success(
                            str(item)
                        )

                else:

                    st.caption(
                        "No clearly documented matching skills."
                    )

            with col2:

                st.subheader(
                    "🔎 Matching Keywords"
                )

                matching_keywords = safe_list(
                    result.get(
                        "matching_keywords"
                    )
                )

                if matching_keywords:

                    for item in matching_keywords:

                        st.write(
                            f"• {item}"
                        )

                else:

                    st.caption(
                        "No supported matching keywords found."
                    )

            # ------------------------------------------------
            # MISSING REQUIREMENTS
            # ------------------------------------------------

            st.subheader(
                "⚠️ Missing or Unverified Requirements"
            )

            missing_requirements = safe_list(
                result.get(
                    "missing_or_unverified_requirements"
                )
            )

            if missing_requirements:

                for item in missing_requirements:

                    st.warning(
                        str(item)
                    )

            else:

                st.success(
                    "No missing or unverified requirements "
                    "were identified from the available information."
                )

            # ------------------------------------------------
            # EXPERIENCE ALIGNMENT
            # ------------------------------------------------

            st.subheader(
                "💼 Experience Alignment"
            )

            experience = result.get(
                "experience_alignment",
                {}
            )

            if experience.get(
                "summary",
                ""
            ):

                st.write(
                    experience.get(
                        "summary",
                        ""
                    )
                )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "**Relevant Experience**"
                )

                experience_items = safe_list(
                    experience.get(
                        "relevant_experience"
                    )
                )

                if experience_items:

                    for item in experience_items:

                        st.write(
                            f"• {item}"
                        )

                else:

                    st.caption(
                        "No specific relevant experience documented."
                    )

            with col2:

                st.markdown(
                    "**Areas Needing Clarification**"
                )

                clarification_items = safe_list(
                    experience.get(
                        "areas_needing_clarification"
                    )
                )

                if clarification_items:

                    for item in clarification_items:

                        st.warning(
                            str(item)
                        )

                else:

                    st.caption(
                        "No additional clarification areas identified."
                    )

            # ------------------------------------------------
            # EDUCATION ALIGNMENT
            # ------------------------------------------------

            st.subheader(
                "🎓 Education Alignment"
            )

            education = result.get(
                "education_alignment",
                {}
            )

            st.write(
                education.get(
                    "summary",
                    ""
                )
            )

            education_items = safe_list(
                education.get(
                    "relevant_education"
                )
            )

            for item in education_items:

                st.write(
                    f"• {item}"
                )

            # ------------------------------------------------
            # PROJECT ALIGNMENT
            # ------------------------------------------------

            st.subheader(
                "🛠️ Project Alignment"
            )

            projects = result.get(
                "project_alignment",
                {}
            )

            st.write(
                projects.get(
                    "summary",
                    ""
                )
            )

            project_items = safe_list(
                projects.get(
                    "relevant_projects"
                )
            )

            for item in project_items:

                st.write(
                    f"• {item}"
                )

            # ------------------------------------------------
            # STRENGTHS / GAPS
            # ------------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "💪 Candidate Strengths"
                )

                strengths = safe_list(
                    result.get(
                        "candidate_strengths"
                    )
                )

                if strengths:

                    for item in strengths:

                        st.success(
                            str(item)
                        )

                else:

                    st.caption(
                        "No specific strengths identified."
                    )

            with col2:

                st.subheader(
                    "📌 Development / Gap Areas"
                )

                gap_areas = safe_list(
                    result.get(
                        "development_or_gap_areas"
                    )
                )

                if gap_areas:

                    for item in gap_areas:

                        st.write(
                            f"• {item}"
                        )

                else:

                    st.caption(
                        "No specific gap areas identified."
                    )

            # ------------------------------------------------
            # INTERVIEW FOCUS
            # ------------------------------------------------

            st.subheader(
                "🎯 Interview Focus Areas"
            )

            interview_focus = safe_list(
                result.get(
                    "interview_focus_areas"
                )
            )

            if interview_focus:

                for index, item in enumerate(
                    interview_focus,
                    1
                ):

                    st.write(
                        f"{index}. {item}"
                    )

            else:

                st.caption(
                    "No interview focus areas generated."
                )

            # ------------------------------------------------
            # INTERVIEW QUESTIONS
            # ------------------------------------------------

            st.subheader(
                "❓ Suggested Interview Questions"
            )

            questions = safe_list(
                result.get(
                    "suggested_interview_questions"
                )
            )

            if questions:

                for index, item in enumerate(
                    questions,
                    1
                ):

                    if isinstance(
                        item,
                        dict
                    ):

                        question = item.get(
                            "question",
                            ""
                        )

                        reason = item.get(
                            "reason",
                            ""
                        )

                    else:

                        question = str(item)
                        reason = ""

                    with st.expander(
                        f"{index}. {question}"
                    ):

                        if reason:

                            st.write(
                                "**Why this question may be useful:**"
                            )

                            st.write(
                                reason
                            )

            else:

                st.caption(
                    "No interview questions generated."
                )

            # ------------------------------------------------
            # RECRUITER NOTES
            # ------------------------------------------------

            st.subheader(
                "📝 Recruiter Notes"
            )

            recruiter_notes = safe_list(
                result.get(
                    "recruiter_notes"
                )
            )

            if recruiter_notes:

                for item in recruiter_notes:

                    st.write(
                        f"• {item}"
                    )

            else:

                st.caption(
                    "No recruiter notes generated."
                )

            # ------------------------------------------------
            # VERIFICATION ITEMS
            # ------------------------------------------------

            st.subheader(
                "🔍 Verification Items"
            )

            verification_items = safe_list(
                result.get(
                    "verification_items"
                )
            )

            if verification_items:

                for index, item in enumerate(
                    verification_items,
                    1
                ):

                    st.checkbox(
                        str(item),
                        key=f"verification_{index}_{candidate_name}"
                    )

            else:

                st.caption(
                    "No additional verification items identified."
                )

            # ------------------------------------------------
            # HUMAN REVIEW NOTICE
            # ------------------------------------------------

            st.divider()

            human_review_notice = result.get(
                "human_review_notice",
                ""
            )

            if human_review_notice:

                st.info(
                    human_review_notice
                )

            else:

                st.info(
                    "AI-generated analysis is provided for "
                    "decision support only. Final recruitment "
                    "decisions must be made by a qualified "
                    "human recruiter after reviewing the "
                    "original candidate information."
                )


    # ========================================================
    # RECRUITER DASHBOARD
    # ========================================================

    elif recruiter_module == "📊 Recruiter Dashboard":

        st.header(
            "📊 Recruiter Dashboard"
        )

        candidates = st.session_state.candidate_records

        scores = [
            c.get(
                "ats_match_score",
                0
            )
            for c in candidates
        ]

        average_ats = (
            sum(scores) / len(scores)
            if scores
            else 0
        )

        reviewing = sum(
            c.get("status") == "Reviewing"
            for c in candidates
        )

        interviews = sum(
            c.get("status") == "Interview"
            for c in candidates
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Candidates",
            len(candidates)
        )

        c2.metric(
            "Average ATS",
            f"{average_ats:.1f}"
        )

        c3.metric(
            "Reviewing",
            reviewing
        )

        c4.metric(
            "Interview",
            interviews
        )

        status_order = [
            "New",
            "Reviewing",
            "Interview",
            "On Hold",
            "Archived"
        ]

        status_counts = {

            status:
                sum(
                    c.get("status") == status
                    for c in candidates
                )

            for status in status_order
        }

        st.subheader(
            "Candidate Status"
        )

        st.bar_chart(
            pd.DataFrame(
                {
                    "Candidates":
                        status_counts
                }
            )
        )

        if candidates:

            st.subheader(
                "Candidate Overview"
            )

            st.dataframe(
                build_candidate_dataframe(),
                use_container_width=True
            )

        else:

            st.info(
                "No candidates available yet."
            )


    # ========================================================
    # CANDIDATE PIPELINE
    # ========================================================

    elif recruiter_module == "🔄 Candidate Pipeline":

        st.header(
            "🔄 Candidate Pipeline"
        )

        statuses = [
            "New",
            "Reviewing",
            "Interview",
            "On Hold",
            "Archived"
        ]

        metrics = st.columns(
            len(statuses)
        )

        for index, status in enumerate(
            statuses
        ):

            count = sum(
                c.get("status") == status
                for c in st.session_state.candidate_records
            )

            metrics[index].metric(
                status,
                count
            )

        st.divider()

        columns = st.columns(
            len(statuses)
        )

        for index, status in enumerate(
            statuses
        ):

            with columns[index]:

                st.subheader(
                    status
                )

                for candidate in st.session_state.candidate_records:

                    if candidate.get(
                        "status"
                    ) != status:

                        continue

                    st.markdown(
                        f"**{candidate.get('candidate_name', '')}**"
                    )

                    st.caption(
                        f"ATS: "
                        f"{candidate.get('ats_match_score', 0)}"
                    )

                    hiring_agent = candidate.get(
                        "hiring_agent_analysis",
                        {}
                    ) or {}

                    if hiring_agent:

                        st.caption(
                            f"AI Agent Alignment: "
                            f"{hiring_agent.get('resume_alignment_score', 0)}"
                        )

                    st.caption(
                        f"{candidate.get('job_title', '')} "
                        f"— "
                        f"{candidate.get('company', '')}"
                    )

                    st.divider()

        st.info(
            "Pipeline status represents workflow state. "
            "It is not an AI hiring recommendation."
        )


    # ========================================================
    # CANDIDATE WORKFLOW
    # ========================================================

    elif recruiter_module == "👤 Candidate Workflow":

        st.header(
            "👤 Candidate Workflow"
        )

        search = st.text_input(
            "Search Candidate"
        )

        filtered_candidates = []

        for candidate in st.session_state.candidate_records:

            if not search.strip():

                filtered_candidates.append(
                    candidate
                )

            else:

                search_text = search.lower()

                if (
                    search_text
                    in candidate.get(
                        "candidate_name",
                        ""
                    ).lower()
                    or
                    search_text
                    in candidate.get(
                        "file_name",
                        ""
                    ).lower()
                    or
                    search_text
                    in candidate.get(
                        "job_title",
                        ""
                    ).lower()
                ):

                    filtered_candidates.append(
                        candidate
                    )

        if not filtered_candidates:

            st.info(
                "No candidates found."
            )

        else:

            candidate_options = [
                (
                    f"{c.get('candidate_name', '')} "
                    f"— {c.get('job_title', '')} "
                    f"— {c.get('candidate_id', '')}"
                )
                for c in filtered_candidates
            ]

            selected_label = st.selectbox(
                "Select Candidate",
                candidate_options
            )

            selected_index = candidate_options.index(
                selected_label
            )

            candidate = filtered_candidates[
                selected_index
            ]

            candidate_id = candidate.get(
                "candidate_id"
            )

            st.subheader(
                candidate.get(
                    "candidate_name",
                    ""
                )
            )

            hiring_agent = candidate.get(
                "hiring_agent_analysis",
                {}
            ) or {}

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "ATS Score",
                candidate.get(
                    "ats_match_score",
                    0
                )
            )

            c2.metric(
                "AI Agent Alignment",
                (
                    f"{hiring_agent.get('resume_alignment_score', 0)}/100"
                    if hiring_agent
                    else "Not Available"
                )
            )

            c3.metric(
                "Status",
                candidate.get(
                    "status",
                    ""
                )
            )

            c4.metric(
                "Interview Rating",
                candidate.get(
                    "interview",
                    {}
                ).get(
                    "rating",
                    0
                )
            )

            st.write(
                f"**Position:** "
                f"{candidate.get('job_title', '')}"
            )

            st.write(
                f"**Company:** "
                f"{candidate.get('company', '')}"
            )

            st.write(
                f"**Resume:** "
                f"{candidate.get('file_name', '')}"
            )

            st.divider()

            # ------------------------------------------------
            # WORKFLOW CONTROL
            # ------------------------------------------------

            st.subheader(
                "⚙️ Workflow Management"
            )

            status_options = [
                "New",
                "Reviewing",
                "Interview",
                "On Hold",
                "Archived"
            ]

            current_status = candidate.get(
                "status",
                "New"
            )

            status_index = (
                status_options.index(
                    current_status
                )
                if current_status in status_options
                else 0
            )

            new_status = st.selectbox(
                "Workflow Status",
                status_options,
                index=status_index,
                key=f"status_{candidate_id}"
            )

            next_action = st.text_input(
                "Next Action",
                value=candidate.get(
                    "next_action",
                    ""
                ),
                key=f"next_action_{candidate_id}"
            )

            next_action_date = st.text_input(
                "Next Action Date",
                value=candidate.get(
                    "next_action_date",
                    ""
                ),
                placeholder="YYYY-MM-DD",
                key=f"next_action_date_{candidate_id}"
            )

            # ------------------------------------------------
            # INTERVIEW
            # ------------------------------------------------

            st.subheader(
                "Interview"
            )

            interview = candidate.get(
                "interview",
                {}
            )

            interview_date = st.text_input(
                "Interview Date",
                value=interview.get(
                    "date",
                    ""
                ),
                key=f"interview_date_{candidate_id}"
            )

            interview_time = st.text_input(
                "Interview Time",
                value=interview.get(
                    "time",
                    ""
                ),
                key=f"interview_time_{candidate_id}"
            )

            interview_mode_options = [
                "",
                "Online",
                "Phone",
                "On-site"
            ]

            current_mode = interview.get(
                "mode",
                ""
            )

            interview_mode = st.selectbox(
                "Interview Mode",
                interview_mode_options,
                index=(
                    interview_mode_options.index(
                        current_mode
                    )
                    if current_mode
                    in interview_mode_options
                    else 0
                ),
                key=f"interview_mode_{candidate_id}"
            )

            interviewer = st.text_input(
                "Interviewer",
                value=interview.get(
                    "interviewer",
                    ""
                ),
                key=f"interviewer_{candidate_id}"
            )

            interview_feedback = st.text_area(
                "Interview Feedback",
                value=interview.get(
                    "feedback",
                    ""
                ),
                key=f"feedback_{candidate_id}"
            )

            rating = st.slider(
                "Interview Rating",
                min_value=0,
                max_value=5,
                value=int(
                    interview.get(
                        "rating",
                        0
                    )
                ),
                key=f"rating_{candidate_id}"
            )

            # ------------------------------------------------
            # TAGS / NOTES
            # ------------------------------------------------

            tags_text = st.text_input(
                "Tags",
                value=", ".join(
                    safe_list(
                        candidate.get(
                            "tags",
                            []
                        )
                    )
                ),
                key=f"tags_{candidate_id}"
            )

            notes = st.text_area(
                "Recruiter Notes",
                value=candidate.get(
                    "notes",
                    ""
                ),
                key=f"notes_{candidate_id}"
            )

            # ------------------------------------------------
            # AI SCREENING SUMMARY
            # ------------------------------------------------

            st.subheader(
                "🔍 AI Screening Summary"
            )

            recruiter_summary = candidate.get(
                "recruiter_summary",
                ""
            )

            if recruiter_summary:

                st.info(
                    recruiter_summary
                )

            else:

                st.caption(
                    "No AI Screening summary has been saved "
                    "for this candidate."
                )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "Strengths"
                )

                for item in safe_list(
                    candidate.get(
                        "strengths"
                    )
                ):

                    st.success(
                        str(item)
                    )

            with col2:

                st.subheader(
                    "Gaps"
                )

                for item in safe_list(
                    candidate.get(
                        "gaps"
                    )
                ):

                    st.write(
                        f"• {item}"
                    )

            # ------------------------------------------------
            # AI HIRING AGENT ANALYSIS
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "🤖 AI Hiring Agent Analysis"
            )

            if hiring_agent:

                agent_score = hiring_agent.get(
                    "resume_alignment_score",
                    0
                )

                try:

                    agent_score = int(
                        agent_score
                    )

                except Exception:

                    agent_score = 0

                agent_score = max(
                    0,
                    min(
                        100,
                        agent_score
                    )
                )

                agent_col1, agent_col2 = st.columns(
                    [1, 2]
                )

                with agent_col1:

                    st.metric(
                        "Resume Alignment",
                        f"{agent_score}/100"
                    )

                with agent_col2:

                    st.progress(
                        agent_score / 100
                    )

                    st.caption(
                        "AI Agent alignment with the documented "
                        "job requirements. This is not a hiring "
                        "recommendation."
                    )

                professional_profile = hiring_agent.get(
                    "professional_profile",
                    ""
                )

                if professional_profile:

                    st.subheader(
                        "Professional Profile"
                    )

                    st.info(
                        professional_profile
                    )

                # --------------------------------------------
                # SKILLS / KEYWORDS
                # --------------------------------------------

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown(
                        "**✅ Matching Skills**"
                    )

                    matching_skills = safe_list(
                        hiring_agent.get(
                            "matching_skills"
                        )
                    )

                    if matching_skills:

                        for item in matching_skills:

                            st.success(
                                str(item)
                            )

                    else:

                        st.caption(
                            "No clearly documented matching skills."
                        )

                with col2:

                    st.markdown(
                        "**🔎 Matching Keywords**"
                    )

                    matching_keywords = safe_list(
                        hiring_agent.get(
                            "matching_keywords"
                        )
                    )

                    if matching_keywords:

                        for item in matching_keywords:

                            st.write(
                                f"• {item}"
                            )

                    else:

                        st.caption(
                            "No supported matching keywords found."
                        )

                # --------------------------------------------
                # MISSING REQUIREMENTS
                # --------------------------------------------

                st.subheader(
                    "⚠️ Missing or Unverified Requirements"
                )

                missing_requirements = safe_list(
                    hiring_agent.get(
                        "missing_or_unverified_requirements"
                    )
                )

                if missing_requirements:

                    for item in missing_requirements:

                        st.warning(
                            str(item)
                        )

                else:

                    st.success(
                        "No missing or unverified requirements "
                        "were identified from the available information."
                    )

                # --------------------------------------------
                # EXPERIENCE
                # --------------------------------------------

                st.subheader(
                    "💼 Experience Alignment"
                )

                experience = hiring_agent.get(
                    "experience_alignment",
                    {}
                )

                if experience.get(
                    "summary",
                    ""
                ):

                    st.write(
                        experience.get(
                            "summary",
                            ""
                        )
                    )

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown(
                        "**Relevant Experience**"
                    )

                    for item in safe_list(
                        experience.get(
                            "relevant_experience"
                        )
                    ):

                        st.write(
                            f"• {item}"
                        )

                with col2:

                    st.markdown(
                        "**Areas Needing Clarification**"
                    )

                    for item in safe_list(
                        experience.get(
                            "areas_needing_clarification"
                        )
                    ):

                        st.warning(
                            str(item)
                        )

                # --------------------------------------------
                # EDUCATION
                # --------------------------------------------

                st.subheader(
                    "🎓 Education Alignment"
                )

                education = hiring_agent.get(
                    "education_alignment",
                    {}
                )

                st.write(
                    education.get(
                        "summary",
                        ""
                    )
                )

                for item in safe_list(
                    education.get(
                        "relevant_education"
                    )
                ):

                    st.write(
                        f"• {item}"
                    )

                # --------------------------------------------
                # PROJECTS
                # --------------------------------------------

                st.subheader(
                    "🛠️ Project Alignment"
                )

                projects = hiring_agent.get(
                    "project_alignment",
                    {}
                )

                st.write(
                    projects.get(
                        "summary",
                        ""
                    )
                )

                for item in safe_list(
                    projects.get(
                        "relevant_projects"
                    )
                ):

                    st.write(
                        f"• {item}"
                    )

                # --------------------------------------------
                # STRENGTHS / GAPS
                # --------------------------------------------

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown(
                        "**💪 Candidate Strengths**"
                    )

                    for item in safe_list(
                        hiring_agent.get(
                            "candidate_strengths"
                        )
                    ):

                        st.success(
                            str(item)
                        )

                with col2:

                    st.markdown(
                        "**📌 Development / Gap Areas**"
                    )

                    for item in safe_list(
                        hiring_agent.get(
                            "development_or_gap_areas"
                        )
                    ):

                        st.write(
                            f"• {item}"
                        )

                # --------------------------------------------
                # INTERVIEW FOCUS
                # --------------------------------------------

                st.subheader(
                    "🎯 Interview Focus Areas"
                )

                interview_focus = safe_list(
                    hiring_agent.get(
                        "interview_focus_areas"
                    )
                )

                if interview_focus:

                    for index, item in enumerate(
                        interview_focus,
                        1
                    ):

                        st.write(
                            f"{index}. {item}"
                        )

                else:

                    st.caption(
                        "No interview focus areas generated."
                    )

                # --------------------------------------------
                # QUESTIONS
                # --------------------------------------------

                st.subheader(
                    "❓ Suggested Interview Questions"
                )

                questions = safe_list(
                    hiring_agent.get(
                        "suggested_interview_questions"
                    )
                )

                if questions:

                    for index, item in enumerate(
                        questions,
                        1
                    ):

                        if isinstance(
                            item,
                            dict
                        ):

                            question = item.get(
                                "question",
                                ""
                            )

                            reason = item.get(
                                "reason",
                                ""
                            )

                        else:

                            question = str(item)
                            reason = ""

                        with st.expander(
                            f"{index}. {question}"
                        ):

                            if reason:

                                st.write(
                                    "**Why this question may be useful:**"
                                )

                                st.write(
                                    reason
                                )

                else:

                    st.caption(
                        "No interview questions generated."
                    )

                # --------------------------------------------
                # RECRUITER NOTES
                # --------------------------------------------

                st.subheader(
                    "📝 AI Recruiter Notes"
                )

                recruiter_notes = safe_list(
                    hiring_agent.get(
                        "recruiter_notes"
                    )
                )

                if recruiter_notes:

                    for item in recruiter_notes:

                        st.write(
                            f"• {item}"
                        )

                else:

                    st.caption(
                        "No AI recruiter notes generated."
                    )

                # --------------------------------------------
                # VERIFICATION
                # --------------------------------------------

                st.subheader(
                    "🔍 Verification Items"
                )

                verification_items = safe_list(
                    hiring_agent.get(
                        "verification_items"
                    )
                )

                if verification_items:

                    for index, item in enumerate(
                        verification_items,
                        1
                    ):

                        st.checkbox(
                            str(item),
                            key=(
                                f"workflow_verification_"
                                f"{candidate_id}_{index}"
                            )
                        )

                else:

                    st.caption(
                        "No additional verification items identified."
                    )

                # --------------------------------------------
                # HUMAN REVIEW NOTICE
                # --------------------------------------------

                human_review_notice = hiring_agent.get(
                    "human_review_notice",
                    ""
                )

                if human_review_notice:

                    st.info(
                        human_review_notice
                    )

                else:

                    st.info(
                        "AI-generated analysis is provided for "
                        "decision support only. Final recruitment "
                        "decisions remain with the human recruiter."
                    )

                # --------------------------------------------
                # AGENT HISTORY
                # --------------------------------------------

                agent_history = candidate.get(
                    "hiring_agent_history",
                    []
                )

                if agent_history:

                    with st.expander(
                        "🕘 AI Hiring Agent Analysis History"
                    ):

                        st.dataframe(
                            pd.DataFrame(
                                agent_history
                            ),
                            use_container_width=True
                        )

            else:

                st.info(
                    "No AI Hiring Agent analysis has been saved "
                    "for this candidate yet."
                )

            # ------------------------------------------------
            # SCREENING HISTORY
            # ------------------------------------------------

            history = candidate.get(
                "screening_history",
                []
            )

            if history:

                st.divider()

                st.subheader(
                    "Screening History"
                )

                st.dataframe(
                    pd.DataFrame(
                        history
                    ),
                    use_container_width=True
                )

            # ------------------------------------------------
            # SAVE / REMOVE
            # ------------------------------------------------

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                if st.button(
                    "💾 Save Candidate Workflow",
                    key=f"save_candidate_{candidate_id}"
                ):

                    candidate["status"] = new_status

                    candidate["next_action"] = next_action

                    candidate[
                        "next_action_date"
                    ] = next_action_date

                    candidate["interview"] = {

                        "date":
                            interview_date,

                        "time":
                            interview_time,

                        "mode":
                            interview_mode,

                        "interviewer":
                            interviewer,

                        "feedback":
                            interview_feedback,

                        "rating":
                            rating,
                    }

                    candidate["tags"] = [
                        tag.strip()
                        for tag in tags_text.split(",")
                        if tag.strip()
                    ]

                    candidate["notes"] = notes

                    st.success(
                        "Candidate workflow saved."
                    )

            with col2:

                if st.button(
                    "🗑️ Remove Candidate",
                    key=f"remove_candidate_{candidate_id}"
                ):

                    st.session_state.candidate_records = [
                        c
                        for c in st.session_state.candidate_records
                        if c.get(
                            "candidate_id"
                        ) != candidate_id
                    ]

                    st.success(
                        "Candidate removed."
                    )

                    st.rerun()


    # ========================================================
    # FOLLOW-UP ASSISTANT
    # ========================================================

    elif recruiter_module == "📧 Follow-up Assistant":

        st.header(
            "📧 Follow-up Assistant"
        )

        st.caption(
            "Create professional candidate communication "
            "messages using the candidate's workflow data."
        )

        candidates = st.session_state.candidate_records

        if not candidates:

            st.info(
                "Add candidates first."
            )

        else:

            candidate_labels = [
                f"{c.get('candidate_name', '')} "
                f"— "
                f"{c.get('job_title', '')} "
                f"— "
                f"{c.get('candidate_id', '')}"
                for c in candidates
            ]

            selected_label = st.selectbox(
                "Candidate",
                candidate_labels,
                key="followup_candidate"
            )

            selected_index = candidate_labels.index(
                selected_label
            )

            candidate = candidates[
                selected_index
            ]

            purpose = st.selectbox(
                "Message Purpose",
                [
                    "Interview Invitation",
                    "Interview Follow-up",
                    "Application Status Update",
                    "On Hold Update",
                    "Next Steps"
                ],
                key="followup_purpose"
            )

            if st.button(
                "✉️ Generate Message",
                key="followup_generate"
            ):

                st.session_state.followup_result = (
                    generate_followup_message(
                        candidate,
                        purpose
                    )
                )

            if st.session_state.followup_result:

                st.divider()

                st.subheader(
                    "Generated Message"
                )

                st.text_area(
                    "Message",
                    st.session_state.followup_result,
                    height=280,
                    key="followup_message_output"
                )

                st.caption(
                    "Review and personalize the message "
                    "before sending."
                )

                st.success(
                    "Message generated successfully."
                )


    # ========================================================
    # RAG KNOWLEDGE ASSISTANT
    # ========================================================

    elif recruiter_module == "🧠 RAG Knowledge Assistant":

        st.header(
            "🧠 RAG Knowledge Assistant"
        )

        st.caption(
            "Upload hiring documents to create a temporary "
            "knowledge base for factual Q&A."
        )

        knowledge_files = st.file_uploader(
            "Upload Hiring Documents",
            type=["pdf"],
            accept_multiple_files=True,
            key="rag_files"
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "🧠 Build Knowledge Base"
            ):

                if not knowledge_files:

                    st.warning(
                        "Please upload at least one PDF."
                    )

                else:

                    with st.spinner(
                        "Analyzing documents..."
                    ):

                        try:

                            knowledge = analyze_multiple_documents(
                                client,
                                knowledge_files
                            )

                            st.session_state.rag_knowledge = knowledge

                            st.session_state.rag_document_name = (
                                f"{len(knowledge_files)} documents"
                            )

                            st.session_state.rag_answer = None

                            st.success(
                                "Knowledge base created."
                            )

                        except Exception as error:

                            st.error(
                                format_gemini_error(
                                    error
                                )
                            )

        with col2:

            if st.button(
                "🗑️ Clear Knowledge Base"
            ):

                st.session_state.rag_knowledge = None

                st.session_state.rag_document_name = None

                st.session_state.rag_answer = None

                st.rerun()

        knowledge = (
            st.session_state.rag_knowledge
        )

        if knowledge:

            st.divider()

            st.metric(
                "Documents",
                knowledge.get(
                    "document_count",
                    0
                )
            )

            for document in knowledge.get(
                "documents",
                []
            ):

                doc_name = document.get(
                    "document_name",
                    "Document"
                )

                doc_knowledge = document.get(
                    "knowledge",
                    {}
                )

                with st.expander(
                    doc_name
                ):

                    st.write(
                        f"**Type:** "
                        f"{doc_knowledge.get('document_type', '')}"
                    )

                    st.write(
                        f"**Title:** "
                        f"{doc_knowledge.get('document_title', '')}"
                    )

                    st.write(
                        f"**Summary:** "
                        f"{doc_knowledge.get('summary', '')}"
                    )

                    st.subheader(
                        "Key Information"
                    )

                    for item in safe_list(
                        doc_knowledge.get(
                            "key_information"
                        )
                    ):

                        st.write(
                            f"• {item}"
                        )

                    st.subheader(
                        "Requirements"
                    )

                    for item in safe_list(
                        doc_knowledge.get(
                            "requirements"
                        )
                    ):

                        st.write(
                            f"• {item}"
                        )

                    st.subheader(
                        "Policies"
                    )

                    for item in safe_list(
                        doc_knowledge.get(
                            "policies"
                        )
                    ):

                        st.write(
                            f"• {item}"
                        )

                    st.subheader(
                        "Process Steps"
                    )

                    for item in safe_list(
                        doc_knowledge.get(
                            "process_steps"
                        )
                    ):

                        st.write(
                            f"• {item}"
                        )

            st.divider()

            st.subheader(
                "Ask Knowledge Assistant"
            )

            question = st.text_area(
                "Your Question",
                placeholder=(
                    "What are the interview requirements?"
                ),
                height=120
            )

            if st.button(
                "💬 Ask Assistant"
            ):

                if not question.strip():

                    st.warning(
                        "Please enter a question."
                    )

                else:

                    with st.spinner(
                        "Searching knowledge..."
                    ):

                        try:

                            answer = ask_knowledge_assistant(
                                client,
                                knowledge,
                                question
                            )

                            st.session_state.rag_answer = answer

                        except Exception as error:

                            st.error(
                                format_gemini_error(
                                    error
                                )
                            )

            if st.session_state.rag_answer:

                answer = st.session_state.rag_answer

                st.subheader(
                    "Answer"
                )

                st.success(
                    answer.get(
                        "answer",
                        ""
                    )
                )

                st.subheader(
                    "Supporting Information"
                )

                for item in safe_list(
                    answer.get(
                        "supporting_information"
                    )
                ):

                    st.write(
                        f"• {item}"
                    )

                st.subheader(
                    "Source Documents"
                )

                for item in safe_list(
                    answer.get(
                        "source_documents"
                    )
                ):

                    st.write(
                        f"• {item}"
                    )

                st.caption(
                    answer.get(
                        "confidence_note",
                        ""
                    )
                )

                st.info(
                    "Knowledge Assistant answers are based "
                    "only on the uploaded documents. "
                    "Human review remains important."
                )


    # ========================================================
    # HIRING FUNNEL ANALYTICS
    # ========================================================

    elif recruiter_module == "📈 Hiring Funnel Analytics":

        st.header(
            "📈 Hiring Funnel Analytics"
        )

        candidates = st.session_state.candidate_records

        total = len(
            candidates
        )

        status_counts = {

            status:
                sum(
                    c.get("status") == status
                    for c in candidates
                )

            for status in [
                "New",
                "Reviewing",
                "Interview",
                "On Hold",
                "Archived"
            ]
        }

        scores = [
            c.get(
                "ats_match_score",
                0
            )
            for c in candidates
        ]

        ratings = [
            c.get(
                "interview",
                {}
            ).get(
                "rating",
                0
            )
            for c in candidates
            if c.get(
                "interview",
                {}
            ).get(
                "rating",
                0
            ) > 0
        ]

        average_ats = (
            sum(scores) / len(scores)
            if scores
            else 0
        )

        interview_count = status_counts.get(
            "Interview",
            0
        )

        interview_rate = (
            interview_count / total * 100
            if total
            else 0
        )

        average_rating = (
            sum(ratings) / len(ratings)
            if ratings
            else 0
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Candidates",
            total
        )

        c2.metric(
            "Average ATS",
            f"{average_ats:.1f}"
        )

        c3.metric(
            "Interview Rate",
            f"{interview_rate:.1f}%"
        )

        c4.metric(
            "Average Rating",
            f"{average_rating:.1f}/5"
        )

        funnel_df = pd.DataFrame(
            {
                "Stage":
                    list(
                        status_counts.keys()
                    ),

                "Candidates":
                    list(
                        status_counts.values()
                    )
            }
        )

        st.subheader(
            "Hiring Funnel"
        )

        st.bar_chart(
            funnel_df.set_index(
                "Stage"
            )
        )

        if candidates:

            job_rows = []

            grouped = {}

            for candidate in candidates:

                key = (
                    candidate.get(
                        "job_title",
                        ""
                    ),
                    candidate.get(
                        "company",
                        ""
                    )
                )

                grouped.setdefault(
                    key,
                    []
                ).append(
                    candidate
                )

            for (
                job_title,
                company
            ), items in grouped.items():

                item_scores = [
                    c.get(
                        "ats_match_score",
                        0
                    )
                    for c in items
                ]

                interview_items = [
                    c
                    for c in items
                    if c.get(
                        "status"
                    ) == "Interview"
                ]

                job_rows.append({

                    "Job":
                        job_title,

                    "Company":
                        company,

                    "Candidates":
                        len(items),

                    "Interview Stage":
                        len(interview_items),

                    "Average ATS":
                        round(
                            sum(item_scores)
                            / len(item_scores),
                            1
                        ),
                })

            st.subheader(
                "Job-wise Analytics"
            )

            st.dataframe(
                pd.DataFrame(
                    job_rows
                ),
                use_container_width=True
            )


    # ========================================================
    # EXPORT
    # ========================================================

    elif recruiter_module == "📥 Export Workflow Data":

        st.header(
            "📥 Export Workflow Data"
        )

        dataframe = build_candidate_dataframe()

        if dataframe.empty:

            st.info(
                "No candidate data available."
            )

        else:

            st.dataframe(
                dataframe,
                use_container_width=True
            )

            csv_data = dataframe.to_csv(
                index=False
            )

            st.download_button(
                "📄 Download CSV",
                data=csv_data,
                file_name="candidate_workflow.csv",
                mime="text/csv"
            )

            excel_buffer = io.BytesIO()

            with pd.ExcelWriter(
                excel_buffer,
                engine="openpyxl"
            ) as writer:

                dataframe.to_excel(
                    writer,
                    index=False,
                    sheet_name="Candidates"
                )

            excel_buffer.seek(0)

            st.download_button(
                "📊 Download Excel",
                data=excel_buffer.getvalue(),
                file_name="candidate_workflow.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                )
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Hiring Platform • V4.0 • Gemini-powered Resume Intelligence • Human Decision Support"
)
