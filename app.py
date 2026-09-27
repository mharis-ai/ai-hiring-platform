import os

import streamlit as st

from supabase import Client, create_client


# ============================================================
# CONFIGURATION
# ============================================================

APP_URL = (
    st.secrets.get("APP_URL")
    if hasattr(st, "secrets") and "APP_URL" in st.secrets
    else os.getenv("APP_URL")
)

SUPABASE_URL = (
    st.secrets.get("SUPABASE_URL")
    if hasattr(st, "secrets") and "SUPABASE_URL" in st.secrets
    else os.getenv("SUPABASE_URL")
)

SUPABASE_PUBLISHABLE_KEY = (
    st.secrets.get("SUPABASE_PUBLISHABLE_KEY")
    if hasattr(st, "secrets")
    and "SUPABASE_PUBLISHABLE_KEY" in st.secrets
    else os.getenv("SUPABASE_PUBLISHABLE_KEY")
)


# ============================================================
# CLIENT
# ============================================================

def get_supabase_client() -> Client:
    """
    Create a per-session Supabase client.

    We intentionally disable client-side session persistence here.
    Streamlit session state owns the authenticated session for the
    current user session.
    """

    if not SUPABASE_URL:
        raise RuntimeError(
            "SUPABASE_URL is missing. "
            "Add it to Streamlit Secrets or your .env file."
        )

    if not SUPABASE_PUBLISHABLE_KEY:
        raise RuntimeError(
            "SUPABASE_PUBLISHABLE_KEY is missing. "
            "Add it to Streamlit Secrets or your .env file."
        )

    return create_client(
        SUPABASE_URL,
        SUPABASE_PUBLISHABLE_KEY,
    )


# ============================================================
# SESSION STATE
# ============================================================

def initialize_auth_state():
    """
    Initialize authentication-related Streamlit session state.
    """

    defaults = {
        "auth_access_token": None,
        "auth_refresh_token": None,
        "auth_user": None,
        "auth_profile": None,
        "auth_initialized": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# ERROR HANDLING
# ============================================================

def _friendly_auth_error(error) -> str:
    """
    Convert Supabase/Auth errors into user-friendly messages.
    """

    message = str(error or "").strip()

    lowered = message.lower()

    if "invalid login credentials" in lowered:
        return "Email or password is incorrect."

    if "email not confirmed" in lowered:
        return (
            "Please verify your email address first, "
            "then try logging in again."
        )

    if "user already registered" in lowered:
        return "An account with this email already exists."

    if "password" in lowered and "least" in lowered:
        return "Please choose a stronger password."

    if "rate limit" in lowered:
        return (
            "Too many authentication attempts. "
            "Please wait a little and try again."
        )

    if "network" in lowered or "connection" in lowered:
        return (
            "Unable to connect to the authentication service. "
            "Please try again."
        )

    return message or "Authentication request failed."


# ============================================================
# RESPONSE HELPERS
# ============================================================

def _get_response_user(response):
    """
    Safely extract a user from different Supabase response shapes.
    """

    user = getattr(response, "user", None)

    if user is not None:
        return user

    data = getattr(response, "data", None)

    if isinstance(data, dict):
        return data.get("user")

    return None


def _get_response_session(response):
    """
    Safely extract a session from different Supabase response shapes.
    """

    session = getattr(response, "session", None)

    if session is not None:
        return session

    data = getattr(response, "data", None)

    if isinstance(data, dict):
        return data.get("session")

    return None


def _get_session_value(session, key):
    """
    Safely read values from Supabase session objects.
    """

    if session is None:
        return None

    value = getattr(session, key, None)

    if value is not None:
        return value

    if isinstance(session, dict):
        return session.get(key)

    return None


def _get_user_value(user, key, default=None):
    """
    Safely read values from Supabase user objects.
    """

    if user is None:
        return default

    value = getattr(user, key, None)

    if value is not None:
        return value

    if isinstance(user, dict):
        return user.get(key, default)

    return default


# ============================================================
# SIGN UP
# ============================================================

def sign_up(
    full_name: str,
    email: str,
    password: str,
    role: str,
):
    """
    Create a new Supabase Auth account.

    The user's role and full name are initially stored as Auth
    metadata. The public profile is created after the user has
    an authenticated session.
    """

    full_name = str(full_name or "").strip()
    email = str(email or "").strip().lower()
    password = str(password or "").strip()
    role = str(role or "").strip()

    if not full_name:
        return False, "Please enter your full name.", None

    if not email:
        return False, "Please enter your email address.", None

    if not password:
        return False, "Please enter a password.", None

    if role not in {"Recruiter", "Job Seeker"}:
        return False, "Please select a valid account role.", None

    if len(password) < 8:
        return (
            False,
            "Password must be at least 8 characters long.",
            None,
        )

    try:
        client = get_supabase_client()

        # Let Supabase use the configured Site URL for email confirmation.
        # This avoids passing a redirect URL from Streamlit and prevents
        # redirect allow-list validation issues during signup.
        signup_options = {
            "data": {
                "full_name": full_name,
                "role": role,
            }
        }

        response = client.auth.sign_up(
            {
                "email": email,
                "password": password,
                "options": signup_options,
            }
        )

        user = _get_response_user(response)
        session = _get_response_session(response)

        if session is not None:
            _store_session(session)
            st.session_state.auth_user = user

            profile = ensure_profile(
                user=user,
                full_name=full_name,
                role=role,
            )

            st.session_state.auth_profile = profile

            return (
                True,
                "Account created successfully.",
                user,
            )

        return (
            True,
            (
                "Account created. Please check your email "
                "and confirm your email address before logging in."
            ),
            user,
        )

    except Exception as error:
        return False, _friendly_auth_error(error), None


# ============================================================
# LOGIN
# ============================================================

def sign_in(
    email: str,
    password: str,
):
    """
    Authenticate an existing user.
    """

    email = str(email or "").strip().lower()
    password = str(password or "")

    if not email:
        return False, "Please enter your email address.", None

    if not password:
        return False, "Please enter your password.", None

    try:
        client = get_supabase_client()

        response = client.auth.sign_in_with_password(
            {
                "email": email,
                "password": password,
            }
        )

        user = _get_response_user(response)
        session = _get_response_session(response)

        if session is None:
            return (
                False,
                "Login succeeded without an active session.",
                None,
            )

        _store_session(session)

        if user is None:
            user = get_current_user()

        st.session_state.auth_user = user

        profile = ensure_profile(user=user)

        if profile is None:
            clear_auth_session()

            return (
                False,
                (
                    "Your account was authenticated, "
                    "but your profile could not be loaded."
                ),
                None,
            )

        st.session_state.auth_profile = profile

        return True, "Login successful.", user

    except Exception as error:
        return False, _friendly_auth_error(error), None


# ============================================================
# SESSION STORAGE
# ============================================================

def _store_session(session):
    """
    Store only the user's current session tokens in Streamlit
    session state.
    """

    access_token = _get_session_value(
        session,
        "access_token",
    )

    refresh_token = _get_session_value(
        session,
        "refresh_token",
    )

    st.session_state.auth_access_token = access_token
    st.session_state.auth_refresh_token = refresh_token


# ============================================================
# CURRENT USER
# ============================================================

def get_current_user():
    """
    Verify and return the current authenticated user.

    We use get_user() instead of trusting local session data.
    """

    access_token = st.session_state.get(
        "auth_access_token"
    )

    refresh_token = st.session_state.get(
        "auth_refresh_token"
    )

    if not access_token or not refresh_token:
        return None

    try:
        client = get_supabase_client()

        client.auth.set_session(
            access_token,
            refresh_token,
        )

        response = client.auth.get_user()

        user = getattr(response, "user", None)

        if user is None:
            data = getattr(response, "data", None)

            if isinstance(data, dict):
                user = data.get("user")

        st.session_state.auth_user = user

        return user

    except Exception:
        clear_auth_session()
        return None


# ============================================================
# PROFILE
# ============================================================

def ensure_profile(
    user=None,
    full_name=None,
    role=None,
):
    """
    Create or update the user's profile row.

    The profile id always comes from auth.users.
    """

    if user is None:
        user = get_current_user()

    if user is None:
        return None

    user_id = _get_user_value(user, "id")

    if not user_id:
        return None

    user_metadata = _get_user_value(
        user,
        "user_metadata",
        {},
    )

    if not isinstance(user_metadata, dict):
        user_metadata = {}

    if not full_name:
        full_name = user_metadata.get(
            "full_name",
            "",
        )

    if not role:
        role = user_metadata.get(
            "role",
            "Job Seeker",
        )

    email = _get_user_value(
        user,
        "email",
        "",
    )

    if role not in {"Recruiter", "Job Seeker"}:
        role = "Job Seeker"

    try:
        client = get_supabase_client()

        client.auth.set_session(
            st.session_state.auth_access_token,
            st.session_state.auth_refresh_token,
        )

        response = (
            client
            .table("profiles")
            .upsert(
                {
                    "id": user_id,
                    "full_name": str(full_name or "").strip(),
                    "email": str(email or "").strip().lower(),
                    "role": role,
                },
                on_conflict="id",
            )
            .execute()
        )

        rows = response.data or []

        if rows:
            return rows[0]

        return None

    except Exception:
        return None


def get_current_profile():
    """
    Load the authenticated user's profile.
    """

    user = get_current_user()

    if user is None:
        return None

    user_id = _get_user_value(user, "id")

    if not user_id:
        return None

    try:
        client = get_supabase_client()

        client.auth.set_session(
            st.session_state.auth_access_token,
            st.session_state.auth_refresh_token,
        )

        response = (
            client
            .table("profiles")
            .select("*")
            .eq("id", user_id)
            .single()
            .execute()
        )

        profile = response.data

        st.session_state.auth_profile = profile

        return profile

    except Exception:
        return None


# ============================================================
# LOGOUT
# ============================================================

def sign_out():
    """
    Sign out the current user and clear local authentication
    state.
    """

    try:
        client = get_supabase_client()

        access_token = st.session_state.get(
            "auth_access_token"
        )
        refresh_token = st.session_state.get(
            "auth_refresh_token"
        )

        if access_token and refresh_token:
            try:
                client.auth.set_session(
                    access_token,
                    refresh_token,
                )
                client.auth.sign_out()
            except Exception:
                pass

    finally:
        clear_auth_session()


def clear_auth_session():
    """
    Clear all authentication state from the current Streamlit
    session.
    """

    st.session_state.auth_access_token = None
    st.session_state.auth_refresh_token = None
    st.session_state.auth_user = None
    st.session_state.auth_profile = None
    st.session_state.auth_initialized = False


# ============================================================
# AUTH CHECK
# ============================================================

def is_authenticated() -> bool:
    """
    Return True only when the current Supabase user can be
    verified.
    """

    user = get_current_user()

    return user is not None


# ============================================================
# PASSWORD RESET
# ============================================================

def request_password_reset(email: str):
    """
    Send a password-reset email.
    """

    email = str(email or "").strip().lower()

    if not email:
        return False, "Please enter your email address."

    try:
        client = get_supabase_client()

        # Let Supabase use the configured Site URL for password reset.
        # The Site URL is already configured in Supabase Authentication.
        client.auth.reset_password_for_email(email)

        return (
            True,
            (
                "If an account exists for this email, "
                "a password reset email has been sent."
            ),
        )

    except Exception as error:
        return False, _friendly_auth_error(error)


# ============================================================
# PASSWORD UPDATE
# ============================================================

def update_password(new_password: str):
    """
    Update the password after a valid recovery session.
    """

    new_password = str(new_password or "")

    if len(new_password) < 8:
        return (
            False,
            "Password must be at least 8 characters long.",
        )

    try:
        client = get_supabase_client()

        access_token = st.session_state.get(
            "auth_access_token"
        )
        refresh_token = st.session_state.get(
            "auth_refresh_token"
        )

        if not access_token or not refresh_token:
            return (
                False,
                "Your password reset session has expired.",
            )

        client.auth.set_session(
            access_token,
            refresh_token,
        )

        client.auth.update_user(
            {
                "password": new_password,
            }
        )

        return True, "Password updated successfully."

    except Exception as error:
        return False, _friendly_auth_error(error)


# ============================================================
# ROLE HELPERS
# ============================================================

def get_user_role():
    """
    Return the authenticated user's application role.
    """

    profile = st.session_state.get(
        "auth_profile"
    )

    if not profile:
        profile = get_current_profile()

    if not profile:
        return None

    return profile.get("role")


def get_user_name():
    """
    Return the authenticated user's display name.
    """

    profile = st.session_state.get(
        "auth_profile"
    )

    if not profile:
        profile = get_current_profile()

    if not profile:
        return None

    return profile.get("full_name")
