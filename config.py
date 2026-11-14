"""Konfigurasi provider LLM dan pengaturan runtime.

Semua nilai dibaca dari variabel lingkungan (`.env`) saat dibutuhkan, bukan
saat modul di-import, sehingga modul ini aman diimpor walau API key belum
diisi (misalnya saat menjalankan test).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-flash",
    },
    "mimo": {
        "api_key_env": "MIMO_API_KEY",
        "base_url": "https://api.xiaomimimo.com/v1",
        "default_model": "mimo-v2.6-flash",
    },
    "zai": {
        "api_key_env": "ZAI_API_KEY",
        "base_url": "https://api.z.ai/api/paas/v4",
        "default_model": "glm-5.3",
    },
    "ifm": {
        "api_key_env": "IFM_API_KEY",
        "base_url": "https://api.ifm.ai/v1",
        "default_model": "IFM/K2-Horizon-375B-A23B",
    },
}


@dataclass(frozen=True)
class LLMSettings:
    """Hasil resolusi provider + model yang akan dipakai."""

    provider_name: str
    api_key: str
    base_url: str
    model_name: str
    temperature: float
    max_tokens: int
    timeout: int
    max_retries: int


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=1)
def get_llm_settings() -> LLMSettings:
    """Baca dan validasi konfigurasi provider dari lingkungan.

    Raises:
        ValueError: Jika provider tidak dikenal atau API key kosong.
    """
    provider_name = os.getenv("MODEL_PROVIDER", "ifm").lower()

    if provider_name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Choose one of the following: {', '.join(PROVIDERS)}"
        )

    provider = PROVIDERS[provider_name]
    api_key = os.getenv(provider["api_key_env"])

    if not api_key:
        raise ValueError(
            f"API key is missing. Set {provider['api_key_env']} "
            "in your .env file."
        )

    return LLMSettings(
        provider_name=provider_name,
        api_key=api_key,
        base_url=provider["base_url"],
        model_name=os.getenv("MODEL_NAME") or provider["default_model"],
        temperature=_env_float("MODEL_TEMPERATURE", 0.2),
        max_tokens=_env_int("MODEL_MAX_TOKENS", 512),
        timeout=_env_int("MODEL_TIMEOUT", 60),
        max_retries=_env_int("MODEL_MAX_RETRIES", 2),
    )
