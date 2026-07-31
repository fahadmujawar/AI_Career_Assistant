import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Make sure it's set in your .env file."
        )

    client = genai.Client(api_key=api_key)
    return client