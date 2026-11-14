"""Tests for the context trimming in :func:`app_hybrid.build_model_messages`.

Pemotongan konteks adalah bagian paling halus di ``app_hybrid``: kalau salah,
yang rusak bukan tampilan tapi permintaan yang dikirim ke model (riwayat
terpotong di tengah giliran, atau ``ToolMessage`` tanpa pendahulunya). Test di
sini murni offline -- ``build_model_messages`` hanya menyusun list pesan.

Run with: pytest test_context_trimming.py
"""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

import app_hybrid

# Satu giliran percakapan yang memakai tool: user bertanya, model meminta
# tool, tool menjawab, model merangkum. Empat pesan, satu kesatuan giliran.
TOOL_TURN = [
    HumanMessage(content="berapa 123 kali 45?"),
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculate",
                "args": {"expression": "123 * 45"},
                "id": "call-1",
            }
        ],
    ),
    ToolMessage(content="5535", tool_call_id="call-1"),
    AIMessage(content="Hasilnya 5535."),
]


def conversation(turns: int) -> list:
    """Bangun riwayat ``turns`` giliran lengkap + satu pertanyaan baru.

    Setiap giliran memakai tool (4 pesan). Riwayat selalu ditutup oleh
    HumanMessage, sama seperti kondisi nyata: pertanyaan user baru saja
    ditambahkan ke state sebelum node LLM jalan.
    """
    messages = []

    for index in range(turns):
        messages.extend(
            [
                HumanMessage(content=f"pertanyaan {index}"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculate",
                            "args": {"expression": f"{index} + 1"},
                            "id": f"call-{index}",
                        }
                    ],
                ),
                ToolMessage(content=str(index + 1), tool_call_id=f"call-{index}"),
                AIMessage(content=f"jawaban {index}"),
            ]
        )

    messages.append(HumanMessage(content="pertanyaan baru"))

    return messages


def build(messages: list, **kwargs) -> list:
    state = {"messages": messages, "route": "general"}
    return app_hybrid.build_model_messages(state, "tester", **kwargs)


def test_system_prompt_is_first_and_not_stored_in_state():
    result = build([HumanMessage(content="hai")])

    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)


def test_empty_history_returns_nothing():
    assert build([]) == []


def test_short_history_is_not_trimmed():
    messages = [HumanMessage(content="hai"), AIMessage(content="halo")]

    result = build(messages, max_tokens=2048, max_messages=12)

    assert [type(m).__name__ for m in result[1:]] == [
        "HumanMessage",
        "AIMessage",
    ]


def test_message_cap_keeps_the_newest_messages():
    # Cap 9 pas mengenai batas satu giliran tool penuh, jadi tidak ada pesan
    # yang terbuang oleh start_on.
    result = build(conversation(10), max_tokens=100_000, max_messages=9)

    assert len(result) == 10  # 1 system + 9 riwayat
    assert isinstance(result[1], HumanMessage)
    assert result[-1].content == "pertanyaan baru"
    assert "jawaban 9" in [message.content for message in result]


def test_message_cap_may_yield_fewer_messages_than_the_cap():
    """Batas jumlah adalah batas atas, bukan jaminan jumlah pesan.

    Karena pemotongan jumlah terjadi sebelum ``start_on="human"``, sisa
    potongan yang berada di tengah giliran ikut dibuang. Untuk riwayat yang
    setiap gilirannya memakai tool, cap 4 menyisakan hanya pertanyaan terbaru.
    Itu disengaja: jendela yang lebih pendek selalu aman, jendela yang terpotong
    di tengah giliran tidak.
    """
    result = build(conversation(10), max_tokens=100_000, max_messages=4)

    assert len(result) <= 5
    assert isinstance(result[1], HumanMessage)
    assert result[-1].content == "pertanyaan baru"


def test_window_never_starts_mid_turn():
    """Inti perbaikan: batas jumlah pesan diterapkan *sebelum* start_on.

    Dengan riwayat panjang yang setiap gilirannya memakai tool, jendela hasil
    harus selalu mulai dari HumanMessage. Sebelum perbaikan, slice
    ``[-MAX_RECENT_MESSAGES:]`` dijalankan setelah ``trim_messages`` sehingga
    jendela bisa dimulai dari AIMessage berisi tool_calls.
    """
    result = build(conversation(10), max_tokens=100_000, max_messages=6)

    assert isinstance(result[1], HumanMessage)


def test_no_orphan_tool_message_at_window_start():
    """``ToolMessage`` tanpa ``AIMessage(tool_calls)`` akan ditolak API."""
    for max_messages in range(1, 12):
        result = build(conversation(8), max_tokens=100_000, max_messages=max_messages)

        window = result[1:]

        assert not isinstance(window[0], ToolMessage), (
            f"max_messages={max_messages} menghasilkan jendela yang "
            f"diawali ToolMessage yatim"
        )

        # Setiap ToolMessage harus didahului AIMessage yang memanggil tool.
        for index, message in enumerate(window):
            if isinstance(message, ToolMessage):
                callers = [
                    earlier
                    for earlier in window[:index]
                    if isinstance(earlier, AIMessage)
                    and any(
                        call["id"] == message.tool_call_id
                        for call in earlier.tool_calls
                    )
                ]
                assert callers, f"ToolMessage pada indeks {index} tidak punya induk"


def test_token_cap_also_starts_on_human():
    """Batas token memakai mekanisme start_on yang sama."""
    result = build(conversation(10), max_tokens=60, max_messages=0)

    assert len(result) > 1
    assert isinstance(result[1], HumanMessage)


def test_token_cap_is_looser_than_message_cap():
    """Kedua batas berlaku bersamaan; yang lebih ketat yang kelihatan."""
    history = conversation(10)

    by_messages = build(history, max_tokens=100_000, max_messages=4)
    by_both = build(history, max_tokens=30, max_messages=4)

    assert len(by_both) <= len(by_messages)


def test_temporary_instruction_reaches_the_system_prompt():
    """Instruksi sementara hanya untuk giliran ini, jadi tidak boleh disimpan."""
    plain = build([HumanMessage(content="ceritakan tentang gravitasi")])
    temporary = build(
        [HumanMessage(content="untuk kali ini saja jawab dalam bahasa Inggris")]
    )

    assert "For this response only" in temporary[0].content
    assert "English" in temporary[0].content
    assert "For this response only" not in plain[0].content
    # Riwayat tidak boleh tercemar oleh instruksi sementara.
    assert len(temporary) == 2


def test_profile_is_injected_into_system_prompt():
    import profile_memory

    profile_memory.update_profile("tester", preferred_language="Japanese")

    result = build([HumanMessage(content="hai")])

    assert "Japanese" in result[0].content


def test_limits_come_from_configuration(monkeypatch):
    """Tanpa argumen eksplisit, batas dibaca dari konfigurasi (bukan hardcoded)."""
    monkeypatch.setenv("MAX_RECENT_MESSAGES", "6")

    import config

    assert config.get_context_settings().max_messages == 6

    result = build(conversation(10), max_tokens=100_000)

    assert len(result) <= 7
    assert isinstance(result[1], HumanMessage)


def test_default_limits_match_documented_values():
    import config

    limits = config.get_context_settings()

    assert limits.max_tokens == 2048
    assert limits.max_messages == 12


def test_tool_turn_survives_when_it_fits():
    """Giliran utuh yang muat tidak boleh dipotong sebagian."""
    result = build(TOOL_TURN, max_tokens=2048, max_messages=12)

    assert [type(m).__name__ for m in result[1:]] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "AIMessage",
    ]
