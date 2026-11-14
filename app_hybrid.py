"""Aplikasi LangGraph hybrid: conditional routing + tool calling.

Menggabungkan conditional edges dan tool calling dalam satu graph::

    START -> router -> llm -> [tools -> llm] -> END

Semua komponen berat dipisah ke modul sendiri (:mod:`router`, :mod:`tools`,
:mod:`llm`, :mod:`config`, :mod:`log`), jadi file ini terutama berisi
perakitan graph dan loop CLI-nya.

Fitur tambahan dibanding ``app_conditional.py``:

* tool calling dengan loop ``llm -> tools -> llm``
* long-term memory profil user (:mod:`profile_memory`)
* riwayat percakapan per sesi lewat ``SqliteSaver``
* streaming token ke terminal
* pemotongan konteks otomatis sebelum memanggil model

Jalankan dengan::

    python app_hybrid.py
"""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Literal

from langchain_core.messages import (
    AIMessageChunk,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import (
    count_tokens_approximately,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode

import profile_memory
from config import get_context_settings
from llm import get_llm
from log import get_logger
from router import routing_sample_count, semantic_route
from tools import TOOLS

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm_with_tools():
    """Model yang sudah di-bind tool, dibuat sekali saat pertama dipakai.

    Sengaja tidak dibuat di level modul: mengimpor modul ini (misalnya dari
    test) tidak boleh menuntut API key sudah terisi.
    """
    return get_llm().bind_tools(TOOLS)

# Batas konteks dibaca dari konfigurasi (lihat config.get_context_settings),
# supaya bisa diubah lewat .env. Pemotongan hanya memengaruhi apa yang dilihat
# model; riwayat lengkap tetap tersimpan di checkpointer.

# user_id mengidentifikasi orang (long-term memory, lintas sesi).
# thread_id mengidentifikasi satu percakapan (short-term memory).
DEFAULT_USER_ID = "local-user"

CHECKPOINT_DB = "checkpoints.sqlite"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

RouteType = Literal["general", "coding", "ml"]


class AgentState(MessagesState):
    """State dengan informasi rute hasil klasifikasi semantik."""

    route: RouteType
    route_distance: float


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

# Base prompt per rute. Prompt profil ditambahkan di atasnya.
SYSTEM_PROMPTS: dict[RouteType, str] = {
    "general": "You are a helpful general assistant.",
    "coding": (
        "You are an expert Python and LangGraph developer. "
        "Provide clean, working code examples with explanations."
    ),
    "ml": (
        "You are a machine learning engineer. Explain concepts clearly "
        "and provide practical code examples when relevant."
    ),
}


def build_system_prompt(route: RouteType, user_id: str) -> str:
    """Gabung prompt rute dengan profil long-term user yang tersimpan."""
    parts = [SYSTEM_PROMPTS.get(route, SYSTEM_PROMPTS["general"])]

    profile = profile_memory.get_profile(user_id)

    if profile:
        details = []

        if profile.get("name"):
            details.append(f"- User's name: {profile['name']}")

        if profile.get("preferred_language"):
            details.append(
                f"- Preferred language: {profile['preferred_language']}"
            )

        if profile.get("response_style"):
            details.append(f"- Response style: {profile['response_style']}")

        if details:
            parts.append(
                "Long-term user profile:\n"
                + "\n".join(details)
                + "\nUse this profile when relevant."
            )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def router_node(state: AgentState):
    """Node pertama: tentukan rute dari pertanyaan user."""
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

    return {
        "route": decision.route,
        "route_distance": decision.distance,
    }


def build_model_messages(
    state: AgentState,
    user_id: str,
    *,
    max_tokens: int | None = None,
    max_messages: int | None = None,
) -> list[AnyMessage]:
    """Susun pesan untuk model: system prompt + riwayat yang dipotong.

    Pemotongan hanya memengaruhi apa yang dikirim ke model. State graph
    (dan dengan itu riwayat di checkpointer) tidak diubah, sehingga user
    tetap bisa melihat percakapan penuh.

    Args:
        max_tokens: Batas token untuk riwayat. Default dari konfigurasi.
        max_messages: Batas jumlah pesan. Default dari konfigurasi.
    """
    limits = get_context_settings()

    if max_tokens is None:
        max_tokens = limits.max_tokens

    if max_messages is None:
        max_messages = limits.max_messages

    messages: list[AnyMessage] = list(state["messages"])

    if not messages:
        return []

    # System prompt dibangun ulang setiap panggilan karena profil user bisa
    # berubah di tengah sesi, dan karena prompt tidak boleh tersimpan di
    # riwayat. Instruksi sementara hanya berlaku untuk giliran ini, jadi ia
    # juga harus dihitung ulang, bukan disimpan di state.
    system_prompt = build_system_prompt(
        state.get("route", "general"),
        user_id,
    )

    temporary = profile_memory.detect_temporary_instructions(
        str(messages[-1].content)
    )

    if temporary:
        overrides = ", ".join(f"{k}: {v}" for k, v in temporary.items())
        system_prompt += f"\n\nFor this response only: {overrides}."

    # Batas keras pertama: jumlah pesan. Dipotong sebelum trim_messages,
    # bukan sesudah, supaya aturan start_on di bawah tetap berlaku atas hasil
    # akhir. Kalau urutannya dibalik, slice ini bisa membuat jendela dimulai
    # dari tengah giliran (misalnya AIMessage berisi tool_calls) karena slice
    # tidak mengenal start_on.
    candidates = (
        messages[-max_messages:] if max_messages > 0 else list(messages)
    )

    # strategy="last" menyimpan pesan-pesan terbaru. start_on="human"
    # menjamin hasil potongan dimulai dari awal giliran user, sehingga tidak
    # pernah menyisakan ToolMessage tanpa AIMessage tool_calls pendahulunya
    # (kondisi yang ditolak API). end_on sengaja tidak dipakai: pesan terakhir
    # justru HumanMessage baru yang akan dijawab.
    return [
        SystemMessage(content=system_prompt),
        *trim_messages(
            candidates,
            max_tokens=max_tokens,
            token_counter=count_tokens_approximately,
            strategy="last",
            start_on="human",
            allow_partial=False,
        ),
    ]


def llm_node(state: AgentState, config: RunnableConfig):
    """Node kedua: model menjawab, mungkin dengan meminta tool."""
    user_id = config.get("configurable", {}).get("user_id", DEFAULT_USER_ID)

    response = get_llm_with_tools().invoke(
        build_model_messages(state, user_id)
    )

    return {"messages": [response]}


def route_after_llm(state: AgentState) -> Literal["tools", "__end__"]:
    """Kalau model meminta tool, jalankan tool; kalau tidak, selesai."""
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return END


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """Bangun dan compile graph.

    ``checkpointer`` sengaja berupa parameter, bukan dibuat di level modul,
    supaya file ini bisa diimpor untuk testing tanpa membuka koneksi DB.
    """
    builder = StateGraph(AgentState)

    builder.add_node("router", router_node)
    builder.add_node("llm", llm_node)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "router")
    builder.add_edge("router", "llm")

    # Conditional: tool loop atau selesai.
    builder.add_conditional_edges(
        "llm",
        route_after_llm,
        {"tools": "tools", END: END},
    )

    # Setelah tool dijalankan, kembali ke llm untuk jawaban akhir.
    builder.add_edge("tools", "llm")

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SESSION_COMMANDS = {
    "/new": "start a new conversation (new thread id)",
    "/reset": "start a new conversation and forget stored preferences",
    "/thread": "show the current thread id",
    "/profile": "show stored user profile",
    "/help": "list session commands",
}


def print_help():
    print("\nSession commands:")

    for command, description in SESSION_COMMANDS.items():
        print(f"  {command:<9} {description}")

    print("  /thread <id>  switch to a specific thread id\n")


def stream_answer(graph, messages: list[AnyMessage], config: dict) -> None:
    """Jalankan graph sambil mencetak token dan aktivitas tool."""
    # stream_mode="messages" menghasilkan (chunk, metadata) untuk setiap
    # token dan setiap pesan lengkap dari node.
    for chunk, metadata in graph.stream(
        {"messages": messages},
        config,
        stream_mode="messages",
    ):
        node = metadata.get("langgraph_node", "")

        if isinstance(chunk, ToolMessage):
            content = str(chunk.content)

            if len(content) > 200:
                content = content[:200] + "..."

            print(f"\n[Tool result] {content}")

        elif (
            isinstance(chunk, AIMessageChunk)
            and node == "llm"
            and chunk.content
        ):
            # Token jawaban dicetak langsung tanpa newline. Tool call tidak
            # dicetak dari chunk karena argumennya datang secara pecahan;
            # print_turn_summary membacanya dari state yang sudah lengkap.
            print(chunk.content, end="", flush=True)

    print()


def print_turn_summary(graph, config: RunnableConfig) -> None:
    """Cetak rute dan tool yang dipakai, dibaca dari state terakhir."""
    snapshot = graph.get_state(config)

    values = snapshot.values if snapshot else {}

    route = values.get("route", "general")
    distance = values.get("route_distance", 0.0)

    print(f"[Route: {route} | distance: {distance:.4f}]")

    for message in values.get("messages", [])[-3:]:
        for call in getattr(message, "tool_calls", None) or []:
            print(f"[Tool call] {call['name']}{call['args']}")


def main():
    print("=== LangGraph Hybrid: Conditional Edge + Tool Calling ===")

    # Vector store dibangun sekali di awal supaya biaya memuat model
    # embedding tidak masuk ke waktu respons pertanyaan pertama.
    print(f"Routing samples loaded: {routing_sample_count()}")

    print(f"Tools available: {', '.join(tool.name for tool in TOOLS)}")
    print("Ask about coding, ML, or general topics.")
    print("Type 'exit' to quit. Type '/help' for session commands.\n")

    # Persistent memory untuk profil user (lintas sesi).
    profile_memory.init_profile_db()

    # Satu identitas user untuk eksperimen lokal. Beda dengan thread_id:
    # user_id mengikat long-term memory, thread_id mengikat satu percakapan.
    user_id = DEFAULT_USER_ID

    def make_config(current_thread_id: str) -> dict:
        return {
            "configurable": {
                "thread_id": current_thread_id,
                "user_id": user_id,
            }
        }

    # Checkpointer untuk riwayat percakapan (per thread).
    # Nama file sama dengan yang dibaca inspect_checkpoints.py.
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)

        # Satu thread per jalankan program, supaya riwayat tidak tercampur
        # dengan sesi sebelumnya.
        thread_id = f"session-{uuid.uuid4()}"
        config = make_config(thread_id)

        print(f"Thread: {thread_id}\n")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                command, *arguments = user_input.split(maxsplit=1)
                argument = arguments[0] if arguments else ""

                if command in {"/new", "/reset"}:
                    thread_id = f"session-{uuid.uuid4()}"
                    config = make_config(thread_id)

                    if command == "/reset":
                        profile_memory.reset_profile(user_id)
                        print(f"New thread and profile cleared: {thread_id}\n")
                    else:
                        print(f"New thread: {thread_id}\n")

                elif command == "/thread":
                    if argument:
                        thread_id = argument
                        config = make_config(thread_id)

                    print(f"Thread: {thread_id}\n")

                elif command == "/profile":
                    profile = profile_memory.get_profile(user_id)

                    if profile:
                        print("\nStored profile:")

                        for key, value in profile.items():
                            print(f"  {key}: {value}")
                    else:
                        print("\nNo profile stored yet.")

                    print()

                elif command == "/help":
                    print_help()

                else:
                    print(f"Unknown command: {command}")
                    print_help()

                continue

            try:
                # 1. Simpan preferensi permanen yang user nyatakan.
                outcome = profile_memory.process_profile_preference(
                    user_id,
                    user_input,
                )

                for key, value in outcome["detected_updates"].items():
                    print(f"[Profile Memory] Saved {key}: {value}")

                # 2. Kenali kalimat perkenalan ("nama saya Fajar").
                name = profile_memory.extract_name_introduction(user_input)

                if name:
                    profile_memory.update_profile_name(user_id, name)
                    print(f"[Profile Memory] Saved name: {name}")

                stream_answer(
                    graph,
                    [HumanMessage(content=user_input)],
                    config,
                )

                print_turn_summary(graph, config)
                print()

            except Exception as exc:
                logger.exception("graph invocation failed")
                print(f"\nAn error occurred: {exc}\n")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        # stdin ditutup (mis. output pipa habis) atau Ctrl+C: keluar diam-diam
        # supaya tidak menampilkan traceback yang tidak berguna. Checkpointer
        # sudah tertutup oleh context manager di dalam main().
        print("\nGoodbye!")
