"""Quick script to check remaining Groq free-tier quota from response headers."""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

response = client.chat.completions.with_raw_response.create(
    model="llama-3.3-70b-versatile",
    max_tokens=1,
    messages=[{"role": "user", "content": "hi"}],
)

headers = response.headers
print(f"Remaining requests (daily):  {headers.get('x-ratelimit-remaining-requests', 'n/a')}")
print(f"Remaining tokens   (daily):  {headers.get('x-ratelimit-remaining-tokens', 'n/a')}")
print(f"Request limit resets in:     {headers.get('x-ratelimit-reset-requests', 'n/a')}")
print(f"Token limit resets in:       {headers.get('x-ratelimit-reset-tokens', 'n/a')}")
