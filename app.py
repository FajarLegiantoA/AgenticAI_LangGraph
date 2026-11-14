"""Aplikasi LangGraph paling sederhana: satu node, satu pertanyaan, satu jawaban.

Ini titik masuk untuk memahami project ini. Versi yang lebih lengkap:

* ``app_tools.py``       -> model memanggil tool
* ``app_conditional.py`` -> routing semantik ke beberapa spesialis
* ``app_hybrid.py``      -> semuanya + memori jangka pendek & panjang

Jalankan dengan::

    python app.py
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from config import get_llm_settings
from llm import get_llm
from log import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are an AI assistant that provides clear and concise "
    "answers in English."
)


class GraphState(TypedDict):
    question: str
    answer: str


def call_model(state: GraphState) -> dict:
    response = get_llm().invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", state["question"]),
        ]
    )

    return {"answer": str(response.content)}


builder = StateGraph(GraphState)

builder.add_node("call_model", call_model)

builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

graph = builder.compile()


def main():
    settings = get_llm_settings()

    print("=== LangGraph AI ===")
    print(f"Provider : {settings.provider_name}")
    print(f"Model    : {settings.model_name}")
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
            logger.exception("graph invocation failed")
            print(f"\nAn error occurred: {exc}\n")


if __name__ == "__main__":
    main()
