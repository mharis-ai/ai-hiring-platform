import json
import os
import re
import time

from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL = "gemini-3.8-flash"


def get_model_name():
    model = os.getenv(
        "GEMINI_MODEL",
        DEFAULT_MODEL
    )

    return model.strip()


# ============================================================
# PDF VALIDATION
# ============================================================

def validate_pdf(uploaded_file):

    if uploaded_file is None:
        raise ValueError(
            "No PDF file was provided."
        )

    file_name = getattr(
        uploaded_file,
        "name",
        "uploaded file"
    )

    if not file_name.lower().endswith(".pdf"):
        raise ValueError(
            f"{file_name} is not a PDF file."
        )

    pdf_bytes = uploaded_file.getvalue()

    if not pdf_bytes:
        raise ValueError(
            f"{file_name} is empty."
        )

    max_size = 50 * 1024 * 1024

    if len(pdf_bytes) > max_size:
        raise ValueError(
            f"{file_name} is larger than 50 MB. "
            "Please upload a smaller PDF."
        )

    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError(
            f"{file_name} does not appear to be a valid PDF."
        )

    return pdf_bytes


# ============================================================
# CREATE PDF PART
# ============================================================

def create_pdf_part(uploaded_file):

    pdf_bytes = validate_pdf(
        uploaded_file
    )

    return types.Part.from_bytes(
        data=pdf_bytes,
        mime_type="application/pdf"
    )


# ============================================================
# RESPONSE TEXT
# ============================================================

def get_response_text(response):

    if response is None:
        raise ValueError(
            "Gemini returned an empty response."
        )

    text = getattr(
        response,
        "text",
        None
    )

    if text is None:
        raise ValueError(
            "Gemini returned no text response."
        )

    text = str(text).strip()

    if not text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    return text


# ============================================================
# CLEAN JSON
# ============================================================

def clean_json_response(response_text):

    text = response_text.strip()

    # Remove ```json
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove ```
    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    text = text.strip()

    # If Gemini adds text before JSON,
    # find the JSON object.
    if not text.startswith("{"):

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:

            text = text[
                start:end + 1
            ]

    return text.strip()


# ============================================================
# PARSE JSON
# ============================================================

def parse_json_response(response):

    response_text = get_response_text(
        response
    )

    cleaned = clean_json_response(
        response_text
    )

    try:

        data = json.loads(
            cleaned
        )

    except json.JSONDecodeError as error:

        raise ValueError(
            "Gemini returned invalid JSON.\n\n"
            f"Details: {error}"
        ) from error

    if not isinstance(data, dict):

        raise ValueError(
            "Gemini returned JSON, but the "
            "result was not an object."
        )

    return data


# ============================================================
# TEMPORARY ERROR DETECTION
# ============================================================

def is_temporary_error(error):

    error_text = str(error).lower()

    signals = [
        "429",
        "rate limit",
        "resource exhausted",
        "503",
        "unavailable",
        "high demand",
        "temporarily",
        "timeout",
        "timed out",
        "connection reset",
        "internal server error",
        "500"
    ]

    return any(
        signal in error_text
        for signal in signals
    )


# ============================================================
# GEMINI RESPONSE
# ============================================================

def generate_gemini_response(
    client,
    contents,
    max_retries=3
):

    model_name = get_model_name()

    last_error = None

    for attempt in range(
        max_retries
    ):

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            return response

        except Exception as error:

            last_error = error

            if not is_temporary_error(
                error
            ):

                raise RuntimeError(
                    f"Gemini API error using "
                    f"model '{model_name}': "
                    f"{error}"
                ) from error

            if attempt == max_retries - 1:
                break

            wait_seconds = 2 ** attempt

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        f"Gemini temporarily failed after "
        f"{max_retries} attempts.\n\n"
        f"Model: {model_name}\n"
        f"Last error: {last_error}"
    )


# ============================================================
# GEMINI JSON
# ============================================================

def generate_gemini_json(
    client,
    contents,
    max_retries=3
):

    response = generate_gemini_response(
        client=client,
        contents=contents,
        max_retries=max_retries
    )

    return parse_json_response(
        response
    )


# ============================================================
# FRIENDLY ERROR
# ============================================================

def format_gemini_error(error):

    error_text = str(error)

    if "404" in error_text:

        return (
            "Gemini model was not found. "
            "Please check GEMINI_MODEL "
            "in your .env file."
        )

    if (
        "401" in error_text
        or "403" in error_text
    ):

        return (
            "Gemini API authentication failed. "
            "Please check your GOOGLE_API_KEY."
        )

    if "429" in error_text:

        return (
            "Gemini API rate limit was reached. "
            "Please wait a moment and try again."
        )

    if "503" in error_text:

        return (
            "Gemini is temporarily unavailable. "
            "Please try again shortly."
        )

    if "invalid json" in error_text.lower():

        return (
            "Gemini returned an unexpected "
            "response format. Please try again."
        )

    return (
        "The AI request could not be completed.\n\n"
        f"Technical details: {error_text}"
    )
