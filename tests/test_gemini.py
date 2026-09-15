import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.ai_client import get_gemini_client

client = get_gemini_client()

response = client.models.generate_content(
    model="gemini-flash-latest",
    contents="Say hello in one short sentence."
)

print(response.text)
