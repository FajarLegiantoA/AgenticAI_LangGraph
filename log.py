"""Inisialisasi logging terpusat.

Level logging dikontrol lewat variabel lingkungan ``LOG_LEVEL``
(default: ``INFO``). Gunakan ``LOG_LEVEL=DEBUG`` untuk melihat detail
router, profil pengguna, dan prompt yang dikirim ke model.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys

_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_console() -> None:
    """Paksa stdout/stderr ke UTF-8.

    Konsol Windows sering memakai codepage non-UTF-8, sehingga karakter
    seperti ``×`` atau ``—`` pada jawaban model tampil sebagai ``?``.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)

        if reconfigure is None:
            continue

        # Stream yang sudah ditutup atau tidak mendukung reconfigure
        # dibiarkan apa adanya; output tetap jalan, hanya encoding-nya lama.
        with contextlib.suppress(OSError, ValueError):
            reconfigure(encoding="utf-8")


def setup_logging(force: bool = False) -> None:
    """Konfigurasikan root logger sekali saja per proses."""
    global _CONFIGURED

    if _CONFIGURED and not force:
        return

    configure_console()

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(level=level, format=_FORMAT, force=force)

    # Perpustakaan ini sangat berisik pada level DEBUG.
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Ambil logger setelah konfigurasi dasar dipastikan aktif."""
    setup_logging()
    return logging.getLogger(name)
