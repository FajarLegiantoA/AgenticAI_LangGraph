
from app import llm
from langchain_core.messages import HumanMessage

response = llm.invoke([
    HumanMessage(content="Reply briefly with: hello")
])

print("Response type:", type(response).__name__)
print("Additional kwargs keys:", list(response.additional_kwargs.keys()))
print("Response metadata keys:", list(response.response_metadata.keys()))

for key in ("reasoning", "reasoning_content"):
    value = response.additional_kwargs.get(key)
    print(f"{key} present in additional_kwargs:", key in response.additional_kwargs)
    print(f"{key} type:", type(value).__name__)
    print(f"{key} is None:", value is None)
    print(f"{key} is empty:", value == "")
