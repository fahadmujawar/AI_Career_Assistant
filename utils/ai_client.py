import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Make sure it's set in your .env file "
            "(local) or in Streamlit Cloud secrets (deployed)."
        )

    client = genai.Client(api_key=api_key)
    return client