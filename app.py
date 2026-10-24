
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END


# 1. Load configuration from .env file
load_dotenv()


# 2. Define configuration for each provider
PROVIDERS = {
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-flash",
    },
    "mimo": {
        "api_key_env": "MIMO_API_KEY",
        "base_url": "https://api.xiaomimimo.com/v1",
        "default_model": "mimo-v2.6-flash",
    },
    "zai": {
        "api_key_env": "ZAI_API_KEY",
        "base_url": "https://api.z.ai/api/paas/v4",
        "default_model": "glm-5.3",
    },
    "ifm": {
        "api_key_env": "IFM_API_KEY",
        "base_url": "https://api.ifm.ai/v1",
        "default_model": "IFM/K2-Horizon-375B-A23B",
    },
}


# 3. Retrieve selected provider
provider_name = os.getenv("MODEL_PROVIDER", "ifm").lower()

if provider_name not in PROVIDERS:
    raise ValueError(
        f"Unknown provider '{provider_name}'. "
        f"Choose one of the following: {', '.join(PROVIDERS)}"
    )

provider = PROVIDERS[provider_name]
api_key = os.getenv(provider["api_key_env"])

if not api_key:
    raise ValueError(
        f"API key is missing. Set {provider['api_key_env']} "
        "in your .env file."
    )

model_name = os.getenv("MODEL_NAME") or provider["default_model"]


class IFMChatOpenAI(ChatOpenAI):
    """Preserve IFM's nonstandard `reasoning` field across turns."""

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(
            response,
            generation_info,
        )

        # Recover the original provider response before LangChain
        # discards nonstandard message fields.
        if isinstance(response, dict):
            raw_response = response
        else:
            raw_response = response.model_dump()

        choices = raw_response.get("choices") or []

        for choice, generation in zip(
            choices,
            result.generations,
        ):
            raw_message = choice.get("message") or {}

            # Preserve the field even when its value is "".
            if (
                isinstance(generation.message, AIMessage)
                and "reasoning" in raw_message
            ):
                generation.message.additional_kwargs["reasoning"] = (
                    raw_message["reasoning"]
                )

        return result

    def _get_request_payload(
        self,
        input_,
        *,
        stop=None,
        **kwargs,
    ):
        payload = super()._get_request_payload(
            input_,
            stop=stop,
            **kwargs,
        )

        # This adapter targets the Chat Completions payload.
        if "messages" not in payload:
            return payload

        input_messages = self._convert_input(input_).to_messages()

        # Reinsert the exact field into the serialized assistant message.
        for message, serialized in zip(
            input_messages,
            payload["messages"],
        ):
            if (
                isinstance(message, AIMessage)
                and "reasoning" in message.additional_kwargs
            ):
                serialized["reasoning"] = (
                    message.additional_kwargs["reasoning"]
                )

        return payload



# 4. Initialize model client

ModelClient = IFMChatOpenAI if provider_name == "ifm" else ChatOpenAI

llm = ModelClient(
    model=model_name,
    api_key=api_key,
    base_url=provider["base_url"],
    temperature=0.2,
    max_tokens=512,
    timeout=60,
    max_retries=2,
)



# 5. Define state structure
class GraphState(TypedDict):
    question: str
    answer: str


# 6. Define node to invoke the AI model
def call_model(state: GraphState) -> dict:
    response = llm.invoke(
        [
            (
                "system",
                "You are an AI assistant that provides clear and concise "
                "answers in English.",
            ),
            ("human", state["question"]),
        ]
    )

    return {"answer": str(response.content)}


# 7. Build graph
builder = StateGraph(GraphState)

builder.add_node("call_model", call_model)

builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

graph = builder.compile()


# 8. Run terminal application
def main():
    print("=== LangGraph AI ===")
    print(f"Provider : {provider_name}")
    print(f"Model    : {model_name}")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("See you later!")
            break

        if not question:
            continue

        try:
            result = graph.invoke({"question": question})
            print(f"\nAI: {result['answer']}\n")
        except Exception as exc:
            print(f"\nAn error occurred: {exc}\n")


if __name__ == "__main__":
    main()
