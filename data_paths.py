"""Penyelesaian path untuk file SQLite di seluruh aplikasi.

Dua aplikasi (CLI ``app_hybrid.py`` dan HTTP ``api_server.py``) memakai
modul ini supaya satu env var, ``DATA_DIR``, memindahkan semua file state
bareng-bareng ke satu direktori:

* ``checkpoints.sqlite``     — riwayat percakapan (checkpointer)
* ``profile_memory.sqlite``  — profil user (long-term memory)

Sesuai prinsip **reliability** (satu lokasi backup), **scalability** (volume
tersendiri, tanpa mengubah kode), dan **maintainability** (logika path ada
di satu tempat, tidak tersebar per aplikasi).

Defaultnya tetap folder proyek, sehingga perilaku CLI lama dan seluruh test
tidak berubah. Di container, ``docker-compose.yml`` menyetel
``DATA_DIR=/app/data`` yang di-mount sebagai volume.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def data_path(filename: str) -> Path:
    """Resolve the absolute path of one SQLite file.

    ``DATA_DIR`` relatif diselesaikan terhadap folder proyek; direktori
    target dibuat otomatis supaya pemanggil tidak perlu mengurusnya.
    """
    raw = os.getenv("DATA_DIR")

    if not raw:
        return PROJECT_ROOT / filename

    root = Path(raw).expanduser()

    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()

    root.mkdir(parents=True, exist_ok=True)

    return root / filename
