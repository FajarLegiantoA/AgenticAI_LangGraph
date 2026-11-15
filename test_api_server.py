"""Offline tests untuk HTTP interface ``api_server.py``.

Suite ini tidak menyentuh LLM, model embedding, maupun file SQLite proyek:

* graph di-build dengan ``InMemorySaver`` + model berskrip (lihat
  ``conftest.ScriptedChatModel``) dan dipasang pada ``app.state``;
* ``router.semantic_route`` di-stub supaya vector store tidak dimuat;
* memori profil diarahkan ke file sementara lewat fixture autouse di
  ``conftest.py``.

Yang diuji adalah kontrak HTTP: status code, bentuk body, dan urutan event
SSE — bukan kualitas jawaban model.

Run with: pytest test_api_server.py
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import api_server
import app_hybrid
import router
from conftest import ScriptedChatModel


@pytest.fixture(autouse=True)
def stub_router(monkeypatch):
    """Router semantik tidak boleh memuat model embedding di test ini."""
    decision = router.RouteDecision(
        route="general",
        distance=0.1,
        matched_sample="stubbed",
        neighbors={"general": 0.1},
        below_threshold=True,
        rescued=False,
    )

    monkeypatch.setattr(
        app_hybrid,
        "semantic_route",
        lambda question, **kwargs: decision,
    )

    monkeypatch.setattr(
        router,
        "routing_sample_count",
        lambda: len(router.ROUTING_SAMPLES),
    )

    return decision


@pytest.fixture
def client(monkeypatch):
    """TestClient dengan graph siap pakai (checkpointer di memori)."""
    from fastapi.testclient import TestClient

    model = ScriptedChatModel()(
        [
            AIMessage(content="Halo! Ada yang bisa saya bantu?"),
            AIMessage(content="Sama-sama."),
        ]
    )

    monkeypatch.setattr(app_hybrid, "get_llm_with_tools", lambda: model)

    graph = app_hybrid.build_graph(checkpointer=InMemorySaver())

    api_server.app.state.graph = graph
    api_server.app.state.checkpointer = InMemorySaver()

    with TestClient(api_server.app) as test_client:
        yield test_client

    del api_server.app.state.graph
    del api_server.app.state.checkpointer


def sse_events(body: str) -> list[tuple[str, dict]]:
    """Parse body SSE menjadi daftar ``(event, payload)``."""
    events = []

    for frame in body.strip().split("\n\n"):
        name, data = None, None

        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])

        if name is not None:
            events.append((name, data))

    return events


# ---------------------------------------------------------------------------
# Helper murni
# ---------------------------------------------------------------------------


def test_resolve_thread_id_generates_when_missing():
    first = api_server.resolve_thread_id(None)
    second = api_server.resolve_thread_id("   ")

    assert first.startswith("api-")
    assert first != second


def test_resolve_thread_id_keeps_client_value():
    assert api_server.resolve_thread_id("  abc  ") == "abc"


def test_make_graph_config_carries_thread_and_user():
    config = api_server.make_graph_config("t-1", "u-1")

    assert config == {"configurable": {"thread_id": "t-1", "user_id": "u-1"}}


def test_sse_serializes_pydantic_and_plain_dict():
    result = api_server.TurnResult(
        thread_id="t-1",
        route="coding",
        route_distance=0.25,
        answer="oke",
    )

    frame = api_server.sse("summary", result)

    assert frame.startswith("event: summary\ndata: ")
    assert frame.endswith("\n\n")
    assert json.loads(frame.split("data: ", 1)[1])["thread_id"] == "t-1"

    # ensure_ascii dimatikan: teks non-ASCII tidak di-escape.
    assert "kopi" in api_server.sse("token", {"content": "kopi"})


def test_tool_calls_of_tolerates_messages_without_calls():
    assert api_server._tool_calls_of(HumanMessage(content="hai")) == []



# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def test_health_reports_graph_ready(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "graph_ready": True}


def test_chat_returns_answer_and_thread_id(client):
    response = client.post("/chat", json={"message": "hai"})

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] == "Halo! Ada yang bisa saya bantu?"
    assert body["route"] == "general"
    assert body["tool_calls"] == []
    assert body["thread_id"]


def test_chat_reuses_thread_for_followup(client):
    first = client.post("/chat", json={"message": "hai"}).json()
    second = client.post(
        "/chat",
        json={"message": "makasih", "thread_id": first["thread_id"]},
    ).json()

    assert second["thread_id"] == first["thread_id"]
    assert second["answer"] == "Sama-sama."


def test_chat_validates_empty_message(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_chat_reports_saved_profile_updates(client):
    body = client.post(
        "/chat",
        json={"message": "mulai sekarang jawab dalam bahasa inggris"},
    ).json()

    assert body["profile_updates"].get("preferred_language") == "English"


def test_chat_reports_saved_profile_name(client):
    body = client.post(
        "/chat",
        json={"message": "nama saya Fajar"},
    ).json()

    assert body["profile_updates"].get("name") == "Fajar"


def test_chat_stream_emits_start_then_summary(client):
    response = client.post("/chat/stream", json={"message": "hai"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = sse_events(response.text)

    assert events[0][0] == "start"
    assert events[0][1]["thread_id"]
    assert events[-1][0] == "summary"
    assert events[-1][1]["answer"] == "Halo! Ada yang bisa saya bantu?"
    assert events[-1][1]["route"] == "general"


def test_chat_stream_reports_error_event(client, monkeypatch):
    """Kegagalan graph harus tertutup dengan event ``error``, bukan 500."""

    def explode(*args, **kwargs):
        raise RuntimeError("model mati")

    monkeypatch.setattr(api_server, "prepare_turn", explode)

    response = client.post("/chat/stream", json={"message": "hai"})

    assert response.status_code == 200

    events = sse_events(response.text)

    assert events[0][0] == "start"
    assert events[-1][0] == "error"
    assert "model mati" in events[-1][1]["detail"]


def test_thread_history_roundtrip(client):
    thread_id = client.post(
        "/chat", json={"message": "hai"}
    ).json()["thread_id"]

    history = client.get(f"/threads/{thread_id}")

    assert history.status_code == 200
    assert [message["role"] for message in history.json()] == ["human", "ai"]
    assert history.json()[1]["content"] == "Halo! Ada yang bisa saya bantu?"


def test_thread_history_unknown_thread_is_404(client):
    assert client.get("/threads/tidak-ada").status_code == 404


def test_delete_thread_clears_history(client):
    thread_id = client.post(
        "/chat", json={"message": "hai"}
    ).json()["thread_id"]

    assert client.delete(f"/threads/{thread_id}").json() == {
        "deleted": thread_id
    }
    assert client.get(f"/threads/{thread_id}").status_code == 404



def test_list_threads_reads_checkpointer_table(
    client, monkeypatch, tmp_path
):
    """/threads membaca checkpoint melalui SqliteSaver.list()."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = tmp_path / "checkpoints.sqlite"

    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        # Simpan checkpoint melalui serializer resmi SqliteSaver.
        for thread_id, count in (("thread-a", 2), ("thread-b", 4)):
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                }
            }

            checkpoint = {
                "v": 1,
                "id": f"checkpoint-{thread_id}",
                "ts": "2026-01-01T00:00:00Z",
                "channel_values": {
                    "messages": [{}] * count,
                },
                "channel_versions": {},
                "versions_seen": {},
                "updated_channels": [],
            }

            saver.put(
                config,
                checkpoint,
                {"source": "loop", "step": count},
                {},
            )

    monkeypatch.setattr(
        api_server, "checkpoint_db_path", lambda: str(db_path)
    )

    response = client.get("/threads")

    assert response.status_code == 200

    threads = {
        item["thread_id"]: item["messages"]
        for item in response.json()
    }

    assert threads == {"thread-a": 2, "thread-b": 4}


def test_profile_endpoints_roundtrip(client):
    assert client.get("/profile").json() == {}

    client.post("/chat", json={"message": "nama saya Fajar"})

    assert client.get("/profile").json()["name"] == "Fajar"

    assert client.delete("/profile").json() == {
        "reset": True,
        "user_id": "local-user",
    }
    assert client.get("/profile").json() == {}


def test_endpoints_return_503_before_startup(monkeypatch):
    """Graph belum ada selama lifespan belum selesai -> jangan 500."""
    from fastapi.testclient import TestClient

    monkeypatch.delattr(api_server.app.state, "graph", raising=False)
    monkeypatch.delattr(api_server.app.state, "checkpointer", raising=False)

    # Tanpa context manager, lifespan tidak dijalankan: state tetap kosong.
    cold = TestClient(api_server.app)

    assert cold.get("/threads").status_code == 503
    assert cold.get("/health").json()["graph_ready"] is False
