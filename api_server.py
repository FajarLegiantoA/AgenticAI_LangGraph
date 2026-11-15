"""HTTP interface (FastAPI) ke graph ``app_hybrid``.

Endpoint yang disediakan:

* ``POST /chat``            — satu giliran chat, jawaban lengkap (JSON).
* ``POST /chat/stream``     — giliran yang sama sebagai SSE (token per token).
* ``GET  /threads``         — daftar thread yang tersimpan di checkpointer.
* ``GET  /threads/{id}``    — riwayat satu thread.
* ``DELETE /threads/{id}``  — hapus riwayat satu thread.
* ``GET  /profile``         — profil long-term satu user.
* ``DELETE /profile``       — lupa seluruh profil user.
* ``GET  /health``          — liveness probe (graph sudah siap?).

Checkpointer dan graph dibuat sekali saat startup (``lifespan``) dan ditutup
saat shutdown, supaya satu koneksi SQLite dipakai bersama oleh semua request.
Node LLM bersifat sinkron dan bisa memakan waktu puluhan detik, jadi panggilan
graph dijalankan di thread pool (``run_in_threadpool``) agar event loop tetap
bisa melayani request lain.

Jalankan dengan::

    uvicorn api_server:app --reload
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

import profile_memory
from app_hybrid import (
    DEFAULT_USER_ID,
    build_graph,
    checkpoint_db_path,
    prepare_turn,
)
from log import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model request/response
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Satu giliran chat. ``thread_id`` kosong = thread baru."""

    message: str = Field(min_length=1, max_length=8000)
    thread_id: str | None = None
    user_id: str = DEFAULT_USER_ID


class ToolCall(BaseModel):
    name: str
    args: dict


class TurnResult(BaseModel):
    thread_id: str
    route: str
    route_distance: float
    answer: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    profile_updates: dict[str, str] = Field(default_factory=dict)


class HistoryMessage(BaseModel):
    role: str
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ThreadInfo(BaseModel):
    thread_id: str
    messages: int


class ThreadDetail(BaseModel):
    """The thread's own id (echoed) + its history, per SDK ThreadState design."""

    thread_id: str
    next_nodes: list[str] = Field(default_factory=list)
    messages: list[HistoryMessage] = Field(default_factory=list)


def _tool_calls_of(message) -> list[ToolCall]:
    return [
        ToolCall(name=call["name"], args=call["args"])
        for call in getattr(message, "tool_calls", None) or []
    ]


# ---------------------------------------------------------------------------
# Helper state
# ---------------------------------------------------------------------------

def make_graph_config(thread_id: str, user_id: str) -> dict:
    """Config LangGraph satu giliran (thread per percakapan, user per orang)."""
    return {"configurable": {"thread_id": thread_id, "user_id": user_id}}


def resolve_thread_id(thread_id: str | None) -> str:
    """Pakai thread_id dari client; kalau kosong, buat thread baru.

    Client tidak perlu tahu bentuk id-nya, jadi format ini tidak boleh
    diandalkan di luar server.
    """
    if thread_id and thread_id.strip():
        return thread_id.strip()

    return f"api-{uuid.uuid4()}"


def thread_exists(graph, thread_id: str) -> bool:
    """Apakah thread ini pernah diproses? Dicetak dari riwayat checkpoint."""
    return next(graph.get_state_history(make_graph_config(thread_id, "")), None) is not None


def read_turn_state(graph, config: dict) -> TurnResult:
    """Ambil hasil satu giliran dari state graph terakhir (baris ringkasan CLI)."""
    snapshot = graph.get_state(config)
    values = snapshot.values if snapshot else {}
    messages = list(values.get("messages", []))

    answer = ""

    final_messages = [
        message for message in messages
        if message.type == "ai" and str(message.content) and not _tool_calls_of(message)
    ]

    if final_messages:
        answer = str(final_messages[-1].content)

    tool_calls = [
        call for message in messages[-3:] for call in _tool_calls_of(message)
    ]

    return TurnResult(
        thread_id=config["configurable"]["thread_id"],
        route=values.get("route", "general"),
        route_distance=float(values.get("route_distance") or 0.0),
        answer=answer,
        tool_calls=tool_calls,
    )


# ---------------------------------------------------------------------------
# SSE
# ---------------------------------------------------------------------------

def sse(event: str, data: dict | BaseModel) -> str:
    """Satu frame Server-Sent Events.

    Model Pydantic di-serialize lewat ``model_dump()``; ensure_ascii dimatikan
    supaya teks Indonesia/emoji tidak berubah jadi escape yang sulit dibaca.
    """
    payload = data.model_dump() if isinstance(data, BaseModel) else data

    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Satu giliran chat
# ---------------------------------------------------------------------------

def run_turn(graph, request: ChatRequest) -> TurnResult:
    """Jalankan satu giliran penuh lalu baca hasilnya dari state.

    Sinkron: dipanggil lewat ``run_in_threadpool`` dari endpoint ``/chat``
    karena ``graph.invoke`` bisa memakan waktu lama dan tidak awaitable.
    """
    thread_id = resolve_thread_id(request.thread_id)
    config = make_graph_config(thread_id, request.user_id)

    saved = prepare_turn(request.user_id, request.message)

    graph.invoke(
        {"messages": [HumanMessage(content=request.message)]},
        config,
    )

    # Preferensi yang baru tersimpan ikut dilaporkan supaya client tahu
    # long-term memory-nya berubah di giliran ini.
    return read_turn_state(graph, config).model_copy(
        update={"profile_updates": saved}
    )


def stream_turn(graph, request: ChatRequest) -> Iterator[str]:
    """Giliran yang sama seperti :func:`run_turn`, tapi sebagai SSE.

    Urutan event: ``start`` -> ``profile`` (kalau ada) -> ``tool_result`` /
    ``token`` -> ``summary``. Event ``error`` menggantikan ``summary`` kalau
    graph gagal di tengah jalan, supaya client selalu menutup stream dengan
    jelas.
    """
    thread_id = resolve_thread_id(request.thread_id)
    config = make_graph_config(thread_id, request.user_id)

    yield sse("start", {"thread_id": thread_id})

    try:
        saved = prepare_turn(request.user_id, request.message)

        if saved:
            yield sse("profile", {"saved": saved})

        for chunk, metadata in graph.stream(
            {"messages": [HumanMessage(content=request.message)]},
            config,
            stream_mode="messages",
        ):
            node = metadata.get("langgraph_node", "")

            if isinstance(chunk, ToolMessage):
                content = str(chunk.content)

                yield sse(
                    "tool_result",
                    {
                        "name": getattr(chunk, "name", "tool"),
                        "content": content[:200],
                        "truncated": len(content) > 200,
                    },
                )

            elif (
                isinstance(chunk, AIMessageChunk)
                and node == "llm"
                and chunk.content
            ):
                # Hanya token dari node llm yang dikirim; chunk tool call
                # datang terpecah dan argumennya sudah ada di event summary.
                yield sse("token", {"content": str(chunk.content)})

        yield sse("summary", read_turn_state(graph, config))

    except Exception as exc:
        logger.exception("stream failed")
        yield sse("error", {"detail": str(exc)})


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> Iterator[None]:
    """Buka checkpointer + graph saat startup, tutup saat shutdown.

    Semua request memakai satu checkpointer yang sama; ``SqliteSaver`` dibuat
    dengan ``check_same_thread=False`` di dalam ``from_conn_string``, jadi
    aman dipakai dari thread pool.
    """
    profile_memory.init_profile_db()

    checkpoint_path = checkpoint_db_path()

    logger.info("startup: checkpointer -> %s", checkpoint_path)

    # Muat vector store router sekarang supaya request pertama tidak
    # membayar biaya memuat model embedding.
    from router import routing_sample_count

    logger.info("routing samples loaded: %d", routing_sample_count())

    with SqliteSaver.from_conn_string(checkpoint_path) as checkpointer:
        app.state.checkpointer = checkpointer
        app.state.graph = build_graph(checkpointer=checkpointer)

        yield

    logger.info("shutdown: checkpointer ditutup")


app = FastAPI(
    title="AgenticAI LangGraph API",
    description=(
        "HTTP interface ke graph ``app_hybrid``: routing semantik, tool "
        "calling, memori jangka pendek (checkpointer SQLite) dan jangka "
        "panjang (profil user)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _graph(app_state: FastAPI):
    """Ambil graph dari app.state; 503 kalau lifespan belum selesai."""
    graph = getattr(app_state.state, "graph", None)

    if graph is None:
        raise HTTPException(status_code=503, detail="server belum siap")

    return graph


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=TurnResult)
async def chat(request: ChatRequest) -> TurnResult:
    """Satu giliran chat; mengembalikan jawaban lengkap sekaligus."""
    graph = _graph(app)

    return await run_in_threadpool(run_turn, graph, request)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Giliran chat sebagai SSE.

    Handler ini sengaja sinkron: generator SSE baru dievaluasi saat body
    stream dikonsumsi, jadi menaruhnya di event loop hanya akan memblokir
    request lain selama graph berjalan.
    """
    graph = _graph(app)

    return StreamingResponse(
        stream_turn(graph, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------

def list_memoized_threads(limit: int = 100) -> list[ThreadInfo]:
    """Daftar thread dari API resmi SqliteSaver, terbaru lebih dulu."""
    checkpointer_path = checkpoint_db_path()

    with SqliteSaver.from_conn_string(checkpointer_path) as saver:
        latest: dict[str, int] = {}

        for item in saver.list(None):
            thread_id = item.config["configurable"]["thread_id"]

            # list() mengurutkan checkpoint terbaru lebih dulu.
            if thread_id not in latest:
                channel_values = item.checkpoint.get("channel_values") or {}
                messages = channel_values.get("messages") or []
                latest[thread_id] = len(messages)

                if len(latest) >= limit:
                    break

    return [
        ThreadInfo(thread_id=thread_id, messages=count)
        for thread_id, count in latest.items()
    ]


@app.get("/threads", response_model=list[ThreadInfo])
async def list_threads(limit: int = 20) -> list[ThreadInfo]:
    """Thread yang tersimpan, terbaru lebih dulu."""
    checkpointer = getattr(app.state, "checkpointer", None)

    if checkpointer is None:
        raise HTTPException(status_code=503, detail="server belum siap")

    return await run_in_threadpool(list_memoized_threads, limit)


@app.get("/threads/{thread_id}", response_model=list[HistoryMessage])
def thread_history(thread_id: str) -> list[HistoryMessage]:
    """Riwayat satu thread apa adanya (termasuk pesan tool)."""
    graph = _graph(app)
    snapshot = graph.get_state(make_graph_config(thread_id, DEFAULT_USER_ID))

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="thread tidak ditemukan")

    return [
        HistoryMessage(
            role=message.type,
            content=str(message.content),
            tool_calls=_tool_calls_of(message),
        )
        for message in snapshot.values.get("messages", [])
    ]


@app.delete("/threads/{thread_id}")
def delete_thread(thread_id: str) -> dict:
    """Hapus seluruh riwayat satu thread dari checkpointer."""
    checkpointer = getattr(app.state, "checkpointer", None)

    if checkpointer is None:
        raise HTTPException(status_code=503, detail="server belum siap")

    checkpointer.delete_thread(thread_id)

    return {"deleted": thread_id}

# ---------------------------------------------------------------------------
# Profile (long-term memory)
# ---------------------------------------------------------------------------

@app.get("/profile")
def read_profile(user_id: str = DEFAULT_USER_ID) -> dict:
    """Profil long-term satu user; dict kosong kalau belum ada."""
    return profile_memory.get_profile(user_id) or {}


@app.delete("/profile")
def erase_profile(user_id: str = DEFAULT_USER_ID) -> dict:
    """Lupakan seluruh profil user (setara ``/reset`` di CLI)."""
    profile_memory.reset_profile(user_id)

    return {"reset": True, "user_id": user_id}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Liveness probe; ``graph_ready`` False saat lifespan belum selesai."""
    return {
        "status": "ok",
        "graph_ready": getattr(app.state, "graph", None) is not None,
    }
