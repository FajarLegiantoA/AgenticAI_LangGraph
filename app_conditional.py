"""Aplikasi LangGraph dengan conditional edge (routing).

Ini versi paling sederhana dari routing: satu node pemilih rute, dua node
tujuan. Rute dihitung oleh modul :mod:`router` (embedding + nearest
neighbour), bukan oleh LLM, jadi murah, cepat, dan bisa diuji.

Jalankan dengan::

    python app_conditional.py
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from llm import get_llm
from log import get_logger
from router import RouteDecision, routing_sample_count, semantic_route

logger = get_logger(__name__)

# Tipe state tambahan: rute terpilih disimpan supaya bisa dibaca node lain.
RouteType = Literal["general", "coding"]


@lru_cache(maxsize=1)
def get_chat_model():
    """Model dibuat saat pertama dipakai, bukan saat modul diimpor."""
    return get_llm()


class RoutingState(MessagesState):
    route: RouteType


def route_node(state: RoutingState):
    """Memilih rute berdasarkan pertanyaan user."""
    question = state["messages"][-1].content
    decision = semantic_route(str(question))

    logger.debug(
        "route=%s distance=%.4f rescued=%s neighbors=%s matched=%r",
        decision.route,
        decision.distance,
        decision.rescued,
        decision.neighbors,
        decision.matched_sample,
    )

    return {"route": decision.route}


def call_general(state: RoutingState):
    messages = [
        SystemMessage(content="You are a helpful general assistant."),
        *state["messages"],
    ]

    response = get_chat_model().invoke(messages)

    return {"messages": [response]}


def call_coding(state: RoutingState):
    messages = [
        SystemMessage(
            content=(
                "You are an expert Python and LangGraph developer. "
                "Provide clean, working code examples with explanations."
            )
        ),
        *state["messages"],
    ]

    response = get_chat_model().invoke(messages)

    return {"messages": [response]}


def route_after_classifier(
    state: RoutingState,
) -> Literal["general_agent", "coding_agent"]:
    """Function yang menentukan node tujuan berikutnya.

    Conditional edge menerima state lalu mengembalikan nama node tujuan.
    """
    return "general_agent" if state["route"] == "general" else "coding_agent"


builder = StateGraph(RoutingState)

builder.add_node("router", route_node)
builder.add_node("general_agent", call_general)
builder.add_node("coding_agent", call_coding)

builder.add_edge(START, "router")

# Router memilih salah satu dari dua agent.
builder.add_conditional_edges(
    "router",
    route_after_classifier,
    {
        "general_agent": "general_agent",
        "coding_agent": "coding_agent",
    },
)

builder.add_edge("general_agent", END)
builder.add_edge("coding_agent", END)

graph = builder.compile()


def show_routing_info(question: str) -> RouteDecision:
    """Tampilkan hasil routing sebelum graph dijalankan."""
    decision = semantic_route(question)

    print(f"\nRouting: {decision.route}")
    print(f"  distance = {decision.distance:.4f}")
    print(f"  rescued  = {decision.rescued}")
    print(f"  neighbors= {decision.neighbors}")

    if decision.matched_sample != "None":
        print(f"  closest  = {decision.matched_sample!r}")

    return decision


def main():
    print("=== LangGraph Conditional Edge (semantic routing) ===")

    # Diagram dicetak di dalam main(), bukan di level modul, supaya
    # mengimpor modul ini untuk testing tidak ikut membangun diagram.
    print(graph.get_graph().draw_mermaid())

    # Bangun vector store sekali di awal supaya waktu routing tidak
    # tercampur dengan waktu pemrosesan pertanyaan.
    print(f"Routing samples loaded: {routing_sample_count()}")

    print("Type a question to see it routed to general_agent or coding_agent.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            show_routing_info(question)

            result = graph.invoke({
                "messages": [HumanMessage(content=question)]
            })

            # Pesan terakhir adalah jawaban agent, pesan sebelumnya adalah
            # hasil dari node router (kalau ada).
            answer = result["messages"][-1]

            print(f"\nAI: {answer.content}\n")

        except Exception as exc:
            logger.exception("graph invocation failed")
            print(f"\nAn error occurred: {exc}\n")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        # stdin ditutup (mis. output pipa habis) atau Ctrl+C: keluar diam-diam
        # supaya tidak menampilkan traceback yang tidak berguna.
        print("\nGoodbye!")
