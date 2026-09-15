import os

from dotenv import load_dotenv
from google import genai
from openai import OpenAI

load_dotenv()

# Display order for the model picker; PROVIDERS[0] is the default selection.
PROVIDERS = ["Gemini", "Groq"]

GEMINI_MODEL = "gemini-flash-latest"
GROQ_MODEL = "openai/gpt-oss-20b"


def _get_secret(name):
    """Read a secret from the environment (.env locally), falling back to
    Streamlit Cloud secrets when deployed there.
    """
    value = os.environ.get(name)

    if not value:
        try:
            import streamlit as st
            value = st.secrets.get(name)
        except Exception:
            value = None

    return value


def get_gemini_client():
    api_key = _get_secret("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Make sure it's set in your .env file "
            "(local) or in Streamlit Cloud secrets (deployed)."
        )

    return genai.Client(api_key=api_key)


def get_groq_client():
    api_key = _get_secret("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Get a free key at "
            "https://console.groq.com/keys, then add it to your .env file "
            "(local) or Streamlit Cloud secrets (deployed)."
        )

    # Groq speaks the OpenAI chat-completions API, so the openai SDK just
    # needs pointing at Groq's base URL instead of OpenAI's.
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def is_demo_mode():
    try:
        import streamlit as st
        return st.secrets.get("DEMO_MODE", "false").lower() == "true"
    except Exception:
        return False


def render_provider_selector():
    """Sidebar dropdown to pick which AI model powers the AI features.

    Shared across pages via the "ai_provider" session_state key, so
    picking a model on one page keeps it selected on the others.
    """
    import streamlit as st

    return st.sidebar.selectbox(
        "AI model",
        PROVIDERS,
        key="ai_provider",
        help=(
            "If the selected model is unavailable (rate-limited or "
            "overloaded), the app automatically retries with the other one."
        ),
    )
