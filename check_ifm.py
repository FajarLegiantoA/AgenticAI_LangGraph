from openai import OpenAI

from config import get_llm_settings

settings = get_llm_settings()

client = OpenAI(
    api_key=settings.api_key,
    base_url=settings.base_url,
)

response = client.chat.completions.create(
    model=settings.model_name,
    messages=[
        {"role": "user", "content": "My name is Alex. Reply briefly."}
    ],
    max_tokens=100,
)

message = response.choices[0].message
raw_message = message.model_dump()

print("Provider:", settings.base_url)
print("Model:", settings.model_name)

print("Response fields:", list(raw_message.keys()))

for key in ("reasoning_content", "reasoning"):
    value = raw_message.get(key)

    print(f"{key} present:", key in raw_message)
    print(f"{key} type:", type(value).__name__)
    print(f"{key} is None:", value is None)
    print(f"{key} is empty string:", value == "")