"""Fixture yang dipakai bersama oleh seluruh test.

Dua hal yang dilakukan file ini:

1. Menanam variabel lingkungan dummy **sebelum** modul proyek diimpor, supaya
   ``config.get_llm_settings()`` tidak pernah menuntut ``.env`` yang benar.
   ``load_dotenv()`` tidak menimpa variabel yang sudah ada, jadi nilai di bawah
   yang menang. Hasilnya: seluruh test berjalan offline, tanpa API key dan
   tanpa panggilan jaringan.
2. Menyediakan fixture ``isolated_profile_db`` yang mengarahkan memori profil
   ke file SQLite sementara, supaya test tidak pernah menulis ke database asli.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

# Harus dijalankan sebelum modul proyek apa pun mengimpor config.py.
_DUMMY_ENV = {
    "MODEL_PROVIDER": "ifm",
    "IFM_API_KEY": "test-key-not-real",
    "MODEL_NAME": "test-model",
    "MAX_CONTEXT_TOKENS": "2048",
    "MAX_RECENT_MESSAGES": "12",
}

for _key, _value in _DUMMY_ENV.items():
    os.environ.setdefault(_key, _value)


@pytest.fixture(autouse=True)
def isolated_profile_db(tmp_path, monkeypatch):
    """Arahkan memori profil ke file SQLite buang-sebelum-pakai.

    ``autouse`` supaya tidak ada test yang tanpa sengaja menulis ke
    ``profile_memory.DB_PATH`` yang asli.
    """
    import profile_memory

    monkeypatch.setattr(
        profile_memory,
        "DB_PATH",
        tmp_path / "profile_memory_test.sqlite",
    )
    profile_memory.init_profile_db()


@pytest.fixture(autouse=True)
def reset_llm_caches():
    """Buang cache objek LLM antar test.

    ``get_llm`` dan ``get_llm_with_tools`` memakai ``lru_cache``. Tanpa
    pembersihan ini, test yang menambal (monkeypatch) salah satunya bisa
    mewariskan model palsu ke test berikutnya.
    """
    import app_conditional
    import app_hybrid
    import llm

    caches = (
        llm.get_llm,
        app_hybrid.get_llm_with_tools,
        app_conditional.get_chat_model,
    )

    for cache in caches:
        cache.cache_clear()

    yield

    for cache in caches:
        cache.cache_clear()


class ScriptedChatModel:
    """Pembuat chat model offline yang menjawab sesuai skrip.

    ``langchain_core`` punya ``FakeMessagesListChatModel``, tapi ia melempar
    ``NotImplementedError`` untuk ``bind_tools`` — padahal node LLM di
    ``app_hybrid`` selalu memanggil ``bind_tools``. Subclass di bawah menutupi
    metode itu dan mengembalikan dirinya sendiri, karena tool hanya perlu
    *tampak* terpasang agar grafnya bisa diuji.
    """

    def __call__(
        self,
        responses: list[Any],
        *,
        record: list | None = None,
    ):
        from langchain_core.language_models.fake_chat_models import (
            FakeMessagesListChatModel,
        )

        class _Scripted(FakeMessagesListChatModel):
            def bind_tools(self, tools: Any, **kwargs: Any):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                # Rekam pesan yang benar-benar dikirim ke model, supaya test
                # bisa memeriksa hasil pemotongan konteks.
                if record is not None:
                    record.append(list(messages))
                return super()._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )

        return _Scripted(responses=responses)
