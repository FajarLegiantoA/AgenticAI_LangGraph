
from openai import OpenAI

from config import get_llm_settings

settings = get_llm_settings()

client = OpenAI(
    api_key=settings.api_key,
    base_url=settings.base_url,
)

messages = [
    {
        "role": "user",
        "content": "My name is Alex. Reply briefly.",
    }
]

try:
    # Turn 1
    response1 = client.chat.completions.create(
        model=settings.model_name,
        messages=messages,
        max_tokens=100,
    )

    assistant = response1.choices[0].message

    print("Turn 1 succeeded")
    print("Assistant content available:", assistant.content is not None)
    print("Reasoning is empty:", assistant.reasoning == "")

    # Replay the assistant response, including the exact reasoning field.
    messages.append({
        "role": "assistant",
        "content": assistant.content,
        "reasoning": assistant.reasoning,
    })

    messages.append({
        "role": "user",
        "content": "What is my name?",
    })

    # Turn 2
    response2 = client.chat.completions.create(
        model=settings.model_name,
        messages=messages,
        max_tokens=100,
    )

    print("Turn 2 succeeded")
    print("Answer:", response2.choices[0].message.content)

except Exception as exc:
    print("Test failed:", type(exc).__name__)
    print(str(exc))
