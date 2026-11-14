"""Tests for the long-term profile memory in :mod:`profile_memory`.

Run with: pytest test_profile_preferences.py
"""

import pytest

from profile_memory import (
    detect_profile_preference,
    detect_temporary_instructions,
    get_profile,
    process_profile_preference,
    update_profile,
    update_profile_name,
)

# Fixture `isolated_profile_db` datang dari conftest.py dan bersifat autouse,
# jadi setiap test di file ini otomatis memakai database profil sementara.


def test_update_profile_persists_all_fields():
    update_profile(
        "local-user",
        name="Fajar",
        preferred_language="English",
        response_style="step-by-step",
    )

    profile = get_profile("local-user")

    assert profile is not None
    assert profile["user_id"] == "local-user"
    assert profile["name"] == "Fajar"
    assert profile["preferred_language"] == "English"
    assert profile["response_style"] == "step-by-step"


def test_update_profile_preserves_untouched_fields():
    update_profile_name("local-user", "Fajar")
    update_profile("local-user", preferred_language="Indonesian")

    profile = get_profile("local-user")

    assert profile["name"] == "Fajar"
    assert profile["preferred_language"] == "Indonesian"
    assert profile["response_style"] is None


def test_get_profile_returns_none_for_unknown_user():
    assert get_profile("unknown-user") is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "  "}, "Name cannot be empty."),
        ({"preferred_language": "  "}, "Preferred language cannot be empty."),
        ({"response_style": "  "}, "Response style cannot be empty."),
    ],
)
def test_update_profile_rejects_blank_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        update_profile("local-user", **kwargs)


def test_update_profile_rejects_empty_user_id():
    with pytest.raises(ValueError, match="user_id cannot be empty"):
        update_profile("   ", name="Fajar")


def test_permanent_preference_is_detected_and_saved():
    result = process_profile_preference(
        "local-user",
        "Mulai sekarang jawab dalam bahasa Indonesia secara singkat saja.",
    )

    assert result["detected_updates"] == {
        "preferred_language": "Indonesian",
        "response_style": "concise",
    }
    assert result["profile"]["preferred_language"] == "Indonesian"
    assert result["profile"]["response_style"] == "concise"


def test_preference_without_permanent_marker_is_ignored():
    assert detect_profile_preference("Jawab dalam bahasa Inggris") == {}


def test_temporary_marker_blocks_permanent_preference():
    message = "Untuk jawaban kali ini saja, jawab dalam bahasa Inggris."

    assert detect_profile_preference(message) == {}
    assert detect_temporary_instructions(message) == {
        "preferred_language": "English"
    }


def test_permanent_marker_blocks_temporary_override():
    message = "From now on, respond in English."

    assert detect_profile_preference(message) == {
        "preferred_language": "English"
    }
    assert detect_temporary_instructions(message) == {}


def test_local_short_request_applies_only_to_current_turn():
    message = "Jelaskan cara kerja RAG, jangan terlalu panjang."

    assert detect_profile_preference(message) == {}
    assert detect_temporary_instructions(message) == {
        "response_style": "concise"
    }
