"""Offline tests for the compiled graphs.

Sebelumnya tidak ada satu pun test yang benar-benar menjalankan graph. Test
di sini memakai chat model dari skrip (lihat fixture ``scripted_model`` di
``conftest.py``) sehingga loop ``llm -> tools -> llm`` bisa diperiksa tanpa
API, tanpa embedding, dan tanpa checkpointer asli.

Run with: pytest test_graph.py
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

import app_hybrid
import router


@pytest.fixture(autouse=True)
def stub_router(monkeypatch):
    """Ganti routing semantik supaya test tidak memuat model embedding.

    Routing itu sendiri sudah diuji di ``test_router_eval.py``. Yang diuji di
    sini adalah struktur grafnya.
    """
    import router

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

    return decision


def scripted(responses, record=None):
    """Satu instance model berskrip.

    ``llm_node`` memanggil ``get_llm_with_tools()`` setiap giliran, jadi model
    yang sama harus dipakai bersama supaya posisi skrip maju, bukan reset.
    """
    from conftest import ScriptedChatModel

    return ScriptedChatModel()(responses, record=record)


def patch_model(monkeypatch, responses, record=None, module=None):
    """Pasang model berskrip pada factory yang dipakai node LLM."""
    module = module or app_hybrid
    model = scripted(responses, record)

    factory_name = (
        "get_chat_model" if module is not app_hybrid else "get_llm_with_tools"
    )

    monkeypatch.setattr(module, factory_name, lambda: model)

    return model


def make_graph(monkeypatch, responses, *, checkpointer=None):
    """Compile graph hybrid dengan model yang menjawab sesuai skrip."""
    patch_model(monkeypatch, responses)

    return app_hybrid.build_graph(checkpointer=checkpointer)


def test_plain_answer_without_tools(monkeypatch):
    graph = make_graph(monkeypatch, [AIMessage(content="halo juga")])

    result = graph.invoke(
        {"messages": [HumanMessage(content="hai")]},
        {"configurable": {"user_id": "tester"}},
    )

    assert result["messages"][-1].content == "halo juga"
    assert result["route"] == "general"


def test_route_is_stored_in_state(monkeypatch):
    graph = make_graph(monkeypatch, [AIMessage(content="oke")])

    result = graph.invoke({"messages": [HumanMessage(content="hai")]})

    assert result["route_distance"] == pytest.approx(0.1)


def test_tool_loop_runs_tool_then_answers(monkeypatch):
    """Ini jalur inti app_hybrid: model minta tool, tool jalan, model merangkum."""
    asking = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculate",
                "args": {"expression": "123 * 45"},
                "id": "call-1",
            }
        ],
    )

    graph = make_graph(monkeypatch, [asking, AIMessage(content="Hasilnya 5535.")])

    result = graph.invoke({"messages": [HumanMessage(content="berapa 123 kali 45?")]})

    kinds = [type(message).__name__ for message in result["messages"]]

    assert kinds == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "AIMessage",
    ]

    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert "5535" in str(tool_message.content)
    assert result["messages"][-1].content == "Hasilnya 5535."


def test_tool_error_does_not_break_the_graph(monkeypatch):
    """Tool yang menolak input berbahaya mengembalikan string, bukan exception."""
    asking = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculate",
                "args": {"expression": "__import__('os')"},
                "id": "call-bad",
            }
        ],
    )

    graph = make_graph(monkeypatch, [asking, AIMessage(content="tidak bisa")])

    result = graph.invoke({"messages": [HumanMessage(content="hack")]})

    tool_message = result["messages"][2]

    # Tool menolak dengan string error, bukan exception, sehingga graf tetap
    # bisa lanjut ke jawaban akhir.
    assert str(tool_message.content).startswith("Error:")
    assert result["messages"][-1].content == "tidak bisa"


def test_model_sees_trimmed_history(monkeypatch):
    """Graf benar-benar memakai ``build_model_messages``, bukan seluruh riwayat."""
    record: list = []

    asking = AIMessage(
        content="",
        tool_calls=[
            {"name": "calculate", "args": {"expression": "1 + 1"}, "id": "c1"}
        ],
    )

    patch_model(
        monkeypatch,
        [asking, AIMessage(content="2"), AIMessage(content="selesai")],
        record,
    )

    graph = app_hybrid.build_graph()

    history = [HumanMessage(content="pertanyaan 0")]

    for index in range(1, 9):
        history.append(asking.model_copy())
        history.append(ToolMessage(content="2", tool_call_id="c1"))
        history.append(AIMessage(content=f"jawaban {index}"))

    history.append(HumanMessage(content="pertanyaan baru"))

    graph.invoke({"messages": history}, {"recursion_limit": 30})

    # Panggilan model pertama harus dimulai dari SystemMessage lalu
    # HumanMessage, bukan dari tengah giliran tool.
    first_call = record[0]

    assert type(first_call[0]).__name__ == "SystemMessage"
    assert isinstance(first_call[1], HumanMessage)
    assert len(first_call) <= 1 + 12


def test_checkpointer_keeps_history_across_turns(monkeypatch):
    """Dua panggilan dengan thread yang sama berbagi riwayat."""
    checkpointer = InMemorySaver()

    patch_model(
        monkeypatch,
        [AIMessage(content="jawab satu"), AIMessage(content="jawab dua")],
    )

    graph = app_hybrid.build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "t1", "user_id": "tester"}}

    graph.invoke({"messages": [HumanMessage(content="pertanyaan satu")]}, config)
    second = graph.invoke(
        {"messages": [HumanMessage(content="pertanyaan dua")]}, config
    )

    contents = [str(message.content) for message in second["messages"]]

    assert "pertanyaan satu" in contents
    assert "jawab satu" in contents
    assert "pertanyaan dua" in contents
    assert second["messages"][-1].content == "jawab dua"


def test_separate_threads_do_not_share_history(monkeypatch):
    checkpointer = InMemorySaver()

    patch_model(monkeypatch, [AIMessage(content="oke")])

    graph = app_hybrid.build_graph(checkpointer=checkpointer)

    first = graph.invoke(
        {"messages": [HumanMessage(content="rahasia")]},
        {"configurable": {"thread_id": "a"}},
    )
    second = graph.invoke(
        {"messages": [HumanMessage(content="halo")]},
        {"configurable": {"thread_id": "b"}},
    )

    assert len(first["messages"]) == 2
    assert len(second["messages"]) == 2


def test_graph_has_the_expected_nodes(monkeypatch):
    graph = make_graph(monkeypatch, [AIMessage(content="x")])

    assert set(graph.get_graph().nodes) >= {"router", "llm", "tools", "__start__"}


def test_conditional_app_routes_to_selected_agent(monkeypatch):
    """``app_conditional`` memilih salah satu dari dua node agen."""
    import app_conditional

    monkeypatch.setattr(
        app_conditional,
        "semantic_route",
        lambda question, **kwargs: router.RouteDecision(
            route="coding",
            distance=0.2,
            matched_sample="stubbed",
            neighbors={"coding": 0.2},
            below_threshold=True,
            rescued=False,
        ),
    )
    patch_model(monkeypatch, [AIMessage(content="contoh kode")], module=app_conditional)

    result = app_conditional.graph.invoke(
        {"messages": [HumanMessage(content="buatkan class python")]}
    )

    assert result["route"] == "coding"
    assert result["messages"][-1].content == "contoh kode"
