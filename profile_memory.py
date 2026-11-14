"""Long-term memory: profil pengguna tersimpan di SQLite.

Seluruh pola deteksi didefinisikan **sekali** di bagian atas modul dan dipakai
bersama oleh ``detect_profile_preference`` (perubahan permanen) dan
``detect_temporary_instructions`` (override satu giliran). Sebelumnya daftar
pola ini disalin di kedua fungsi, sehingga menambah satu bahasa/gaya berarti
harus mengubah dua tempat — sumber bug klasik "temporary melubangi permanent".

Aturan main yang dijaga oleh test di ``test_profile_preferences.py``:

* Penanda permanen ("mulai sekarang", "from now on") -> simpan ke profil.
* Penanda sementara ("untuk kali ini saja", "just this once") -> hanya untuk
  giliran ini, tidak disimpan.
* Keduanya muncul bersamaan -> dianggap sementara (yang lebih spesifik menang).
* Permintaan gaya lokal tanpa penanda ("jangan terlalu panjang") -> berlaku
  untuk giliran ini saja, tidak disimpan.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "profile_memory.sqlite"


# =========================================================
# Pola deteksi (satu sumber kebenaran)
# =========================================================

# Menandai instruksi yang hanya berlaku untuk jawaban saat ini.
TEMPORARY_MARKERS: tuple[str, ...] = (
    r"\bfor this answer only\b",
    r"\bfor this time only\b",
    r"\bjust this once\b",
    r"\bfor this response\b",
    r"\buntuk jawaban kali ini saja\b",
    r"\buntuk kali ini saja\b",
    r"\buntuk kali ini\b",
    r"\buntuk jawaban ini saja\b",
    r"\bkhusus jawaban ini\b",
)

# Menandai instruksi yang boleh disimpan permanen ke profil.
PERMANENT_MARKERS: tuple[str, ...] = (
    r"\bfrom now on\b",
    r"\bgoing forward\b",
    r"\bpermanently\b",
    r"\bmulai sekarang\b",
    r"\bseterusnya\b",
    r"\bselalu\b",
)

# Bahasa. Varian "strict" menuntut frasa lengkap supaya perubahan permanen
# tidak terjadi karena kebetulan; varian "loose" lebih longgar karena
# dampaknya hanya satu giliran.
LANGUAGE_PATTERNS: dict[str, dict[str, tuple[str, ...]]] = {
    "English": {
        "strict": (
            r"\banswer in english\b",
            r"\brespond in english\b",
            r"\buse english\b",
            r"\bgunakan bahasa inggris\b",
            r"\bpakai bahasa inggris\b",
            r"\bjawab dalam bahasa inggris\b",
        ),
        "loose": (
            r"\banswer in english\b",
            r"\brespond in english\b",
            r"\buse english\b",
            r"\bbahasa inggris\b",
            r"\bberbahasa inggris\b",
        ),
    },
    "Indonesian": {
        "strict": (
            r"\banswer in indonesian\b",
            r"\brespond in indonesian\b",
            r"\buse indonesian\b",
            r"\bgunakan bahasa indonesia\b",
            r"\bpakai bahasa indonesia\b",
            r"\bjawab dalam bahasa indonesia\b",
        ),
        "loose": (
            r"\banswer in indonesian\b",
            r"\brespond in indonesian\b",
            r"\buse indonesian\b",
            r"\bbahasa indonesia\b",
            r"\bberbahasa indonesia\b",
        ),
    },
}

# Permintaan jawaban singkat punya padanan yang sama di kedua jalur: frasa
# "singkat saja" diperlakukan identik apakah tujuannya permanen atau sementara.
CONCISE_PATTERNS: tuple[str, ...] = (
    r"\bbriefly\b",
    r"\bbe concise\b",
    r"\bshort answers?\b",
    r"\bkeep (?:your )?answers? short\b",
    r"\bkeep it short\b",
    r"\bkeep it brief\b",
    r"\bsingkat saja\b",
    r"\bjawaban singkat\b",
    r"\bjangan terlalu panjang\b",
    r"\bjawab(?:lah)? secara singkat\b",
    r"\bbuat jawaban(?:nya)? singkat\b",
    r"\bjelaskan secara singkat\b",
)

# "step-by-step" hanya menerima "secara bertahap" pada jalur permanen.
STYLE_PATTERNS: dict[str, dict[str, tuple[str, ...]]] = {
    "step-by-step": {
        "strict": (
            r"\bstep by step\b",
            r"\blangkah demi langkah\b",
            r"\bsecara bertahap\b",
        ),
        "loose": (
            r"\bstep by step\b",
            r"\blangkah demi langkah\b",
        ),
    },
    "concise": {
        "strict": CONCISE_PATTERNS,
        "loose": CONCISE_PATTERNS,
    },
}

# Permintaan gaya yang wajar dianggap berlaku satu giliran saja, sehingga tidak
# perlu penanda "untuk kali ini saja".
LOCAL_STYLE_REQUESTS: tuple[str, ...] = (
    r"\bjangan terlalu panjang\b",
    r"\bsingkat saja\b",
    r"\bjawaban singkat\b",
    r"\bjawab(?:lah)? secara singkat\b",
    r"\bbuat jawaban(?:nya)? singkat\b",
    r"\bjelaskan secara singkat\b",
)

# Pola pengenalan nama, dipakai app_hybrid untuk menyimpan nama profil.
NAME_INTRODUCTION = re.compile(
    r"^\s*(?:my name is|nama saya|nama aku)\s+"
    r"([A-Za-z][A-Za-z '\-]{0,79})\s*[.!]?\s*$",
    re.IGNORECASE,
)


def _matches_any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _first_match(
    mapping: dict[str, dict[str, tuple[str, ...]]],
    text: str,
    variant: str,
) -> str | None:
    """Kembalikan kunci pertama (urutan dict) yang polanya cocok."""
    for value, variants in mapping.items():
        if _matches_any(variants[variant], text):
            return value

    return None


def extract_language_and_style(text: str, *, variant: str = "strict") -> dict:
    """Ekstrak bahasa dan gaya jawaban dari satu kalimat.

    Args:
        text: Teks pesan pengguna, sudah di-lower-case.
        variant: ``"strict"`` untuk perubahan permanen, ``"loose"`` untuk
            override sementara.
    """
    updates: dict[str, str] = {}

    language = _first_match(LANGUAGE_PATTERNS, text, variant)

    if language:
        updates["preferred_language"] = language

    style = _first_match(STYLE_PATTERNS, text, variant)

    if style:
        updates["response_style"] = style

    return updates


def _now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# =========================================================
# Penyimpanan
# =========================================================

def init_profile_db():
    """Create the profile table and safely migrate legacy schemas."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                updated_at TEXT NOT NULL
            )
        """)

        # Inspect existing columns to ensure idempotent schema migrations.
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(user_profiles)"
            ).fetchall()
        }

        if "preferred_language" not in columns:
            conn.execute("""
                ALTER TABLE user_profiles
                ADD COLUMN preferred_language TEXT
            """)

        if "response_style" not in columns:
            conn.execute("""
                ALTER TABLE user_profiles
                ADD COLUMN response_style TEXT
            """)


def get_profile(user_id: str) -> dict | None:
    """Retrieve full profile information for a given user."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("""
            SELECT
                user_id,
                name,
                preferred_language,
                response_style,
                updated_at
            FROM user_profiles
            WHERE user_id = ?
        """, (user_id,)).fetchone()

    return dict(row) if row else None


def update_profile(
    user_id: str,
    *,
    name: str | None = None,
    preferred_language: str | None = None,
    response_style: str | None = None,
) -> None:
    """Update only the provided fields while preserving others."""
    if not user_id.strip():
        raise ValueError("user_id cannot be empty.")

    labels = {
        "name": "Name",
        "preferred_language": "Preferred language",
        "response_style": "Response style",
    }

    candidates = {
        "name": name,
        "preferred_language": preferred_language,
        "response_style": response_style,
    }

    updates = {}

    for column, value in candidates.items():
        if value is None:
            continue

        if not value.strip():
            raise ValueError(f"{labels[column]} cannot be empty.")

        updates[column] = value.strip()

    with sqlite3.connect(DB_PATH) as conn:
        # Ensure the profile record exists without overwriting existing data.
        conn.execute("""
            INSERT INTO user_profiles (user_id, updated_at)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO NOTHING
        """, (user_id.strip(), _now()))

        if updates:
            assignments = ", ".join(f"{column} = ?" for column in updates)
            values = list(updates.values())
            values.extend([_now(), user_id.strip()])

            conn.execute(
                f"""
                UPDATE user_profiles
                SET {assignments}, updated_at = ?
                WHERE user_id = ?
                """,
                values,
            )


def update_profile_name(user_id: str, name: str) -> None:
    """Save or update the user's name."""
    name = name.strip()

    if not user_id.strip():
        raise ValueError("user_id cannot be empty.")

    if not name:
        raise ValueError("Name cannot be empty.")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO user_profiles (user_id, name, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                name = excluded.name,
                updated_at = excluded.updated_at
        """, (user_id, name, _now()))


def reset_profile(user_id: str) -> None:
    """Hapus seluruh long-term memory untuk satu user."""
    if not user_id.strip():
        raise ValueError("user_id cannot be empty.")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "DELETE FROM user_profiles WHERE user_id = ?",
            (user_id.strip(),),
        )


def extract_name_introduction(user_message: str) -> str | None:
    """Ambil nama dari kalimat perkenalan, atau None kalau bukan perkenalan."""
    match = NAME_INTRODUCTION.match(user_message)

    if not match:
        return None

    return match.group(1).strip().rstrip(".!").strip() or None


# =========================================================
# Deteksi preferensi
# =========================================================

def detect_profile_preference(user_message: str) -> dict:
    """Detect explicit permanent preference changes.

    Returns an empty dict when the message carries a temporary marker or
    lacks an explicit permanent-change phrase.
    """
    text = user_message.lower()

    # Temporary instructions must not change permanent preferences.
    if _matches_any(TEMPORARY_MARKERS, text):
        return {}

    # Require an explicit permanent-change phrase.
    if not _matches_any(PERMANENT_MARKERS, text):
        return {}

    return extract_language_and_style(text, variant="strict")


def detect_temporary_instructions(user_message: str) -> dict:
    """Detect one-response language and style overrides.

    A permanent marker without a temporary marker is left to
    :func:`detect_profile_preference` so the two paths never fight.
    """
    text = user_message.lower()

    has_temporary_marker = _matches_any(TEMPORARY_MARKERS, text)
    has_permanent_marker = _matches_any(PERMANENT_MARKERS, text)
    has_local_style_request = _matches_any(LOCAL_STYLE_REQUESTS, text)

    # Instruksi permanen ditangani oleh profile memory.
    if has_permanent_marker and not has_temporary_marker:
        return {}

    # Tanpa penanda sementara atau permintaan gaya lokal, tidak ada override.
    if not has_temporary_marker and not has_local_style_request:
        return {}

    return extract_language_and_style(text, variant="loose")


def process_profile_preference(user_id: str, user_message: str) -> dict:
    """Detect explicit preference changes and persist them."""
    updates = detect_profile_preference(user_message)

    if updates:
        update_profile(user_id, **updates)

    return {
        "detected_updates": updates,
        "profile": get_profile(user_id),
    }


if __name__ == "__main__":
    init_profile_db()

    print("Database schema successfully verified/updated.")
    print("Current profile:", get_profile("local-user"))
