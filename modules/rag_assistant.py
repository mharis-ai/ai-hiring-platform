import json

from modules.gemini_utils import (
    create_pdf_part,
    generate_gemini_json,
)


def analyze_knowledge_document(
    client,
    uploaded_file
):

    pdf_part = create_pdf_part(
        uploaded_file
    )

    prompt = """
You are an AI Hiring Knowledge Assistant.

Analyze the uploaded document and extract
useful factual information for a recruitment team.

IMPORTANT RULES:

1. Analyze the complete document.
2. Use ONLY information present in the document.
3. NEVER invent information.
4. NEVER assume missing information.
5. If information is unavailable, say:
   "Information not found in the provided document."
6. Do not make hiring decisions.
7. Return only JSON.

Return exactly:

{
    "document_title": "",
    "document_type": "",
    "summary": "",
    "key_information": [],
    "requirements": [],
    "policies": [],
    "process_steps": [],
    "important_keywords": [],
    "faq": [
        {
            "question": "",
            "answer": ""
        }
    ],
    "source_notes": []
}
"""

    contents = [
        pdf_part,
        prompt
    ]

    return generate_gemini_json(
        client,
        contents
    )


def analyze_multiple_documents(
    client,
    uploaded_files
):

    documents = []

    for uploaded_file in uploaded_files:

        knowledge = analyze_knowledge_document(
            client,
            uploaded_file
        )

        documents.append(
            {
                "document_name": uploaded_file.name,
                "knowledge": knowledge
            }
        )

    return {
        "document_count": len(documents),
        "documents": documents
    }


def ask_knowledge_assistant(
    client,
    knowledge,
    question
):

    knowledge_json = json.dumps(
        knowledge,
        indent=2,
        ensure_ascii=False
    )

    prompt = f"""
You are an AI Hiring Knowledge Assistant.

Answer the user's question using ONLY
the provided hiring knowledge base.

IMPORTANT RULES:

1. Use only the provided knowledge.
2. Never invent information.
3. Never assume missing information.
4. If the answer cannot be found, say:

"This information is not available
in the provided documents."

5. Keep the answer factual and concise.
6. Do not make hiring decisions.
7. Do not say Hire, Reject, or Select.
8. Combine multiple relevant documents carefully.
9. Identify supporting documents whenever possible.
10. Return only JSON.

Return exactly:

{{
    "answer": "",
    "supporting_information": [],
    "source_documents": [],
    "confidence_note": ""
}}

KNOWLEDGE BASE:

{knowledge_json}

USER QUESTION:

{question}
"""

    return generate_gemini_json(
        client,
        prompt
    )
