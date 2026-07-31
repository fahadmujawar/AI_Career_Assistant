from utils.ai_client import get_gemini_client

client = get_gemini_client()

response = client.models.generate_content(
    model="gemini-flash-latest",
    contents="Say hello in one short sentence."
)

print(response.text)