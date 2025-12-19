from openai import OpenAI
import os
from dotenv import load_dotenv

# Force reload of .env file, overwriting any existing env vars
load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

print(f"Debug - API Key: {api_key[:20]}...{api_key[-10:] if api_key else 'None'}")
print(f"Debug - API Base: {api_base}")

client = OpenAI(
    api_key=api_key,
    base_url=api_base,
)

response = client.chat.completions.create(
    model="mistralai/devstral-2512:free",
    messages=[{"role": "user", "content": "Say hello"}],
)

print(response.choices[0].message.content)
