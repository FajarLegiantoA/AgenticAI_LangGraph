"""Aplikasi LangGraph dengan pemanggilan tool.

Graph-nya berbentuk loop: model menjawab, dan kalau ia meminta tool,
``ToolNode`` menjalankannya lalu hasilnya dikembalikan ke model.

Jalankan dengan::

    python app_tools.py
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import HumanMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from llm import get_llm
from log import get_logger
from tools import TOOLS

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm_with_tools():
    """Model + tool dibuat saat pertama dipakai, bukan saat modul diimpor.

    Model harus mendukung tool/function calling untuk provider yang dipilih.
    """
    return get_llm().bind_tools(TOOLS)


def call_model(state: MessagesState):
    response = get_llm_with_tools().invoke(state["messages"])

    # MessagesState otomatis menambahkan pesan baru ke riwayat.
    return {"messages": [response]}


builder = StateGraph(MessagesState)

builder.add_node("llm", call_model)
builder.add_node("tools", ToolNode(TOOLS))

builder.add_edge(START, "llm")

# Kalau model meminta tool, jalankan ToolNode; kalau tidak, akhiri graph.
builder.add_conditional_edges("llm", tools_condition)

# Setelah tool selesai, model menerima hasilnya dan menyusun jawaban akhir.
builder.add_edge("tools", "llm")

graph = builder.compile()


def main():
    print("=== LangGraph Tool Calling ===")
    print("Available tools:", ", ".join(tool.name for tool in TOOLS))
    print("Try: 'what is 23 * 7?' or 'list files in the current directory'")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            result = graph.invoke({
                "messages": [HumanMessage(content=question)]
            })

            # Tampilkan tool yang sebenarnya dipanggil supaya terlihat alurnya.
            for message in result["messages"][1:]:
                for call in getattr(message, "tool_calls", None) or []:
                    print(f"[Tool call] {call['name']}{call['args']}")

            print(f"\nAI: {result['messages'][-1].content}\n")

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
