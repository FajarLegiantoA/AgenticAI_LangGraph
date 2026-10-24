import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parent / "profile_memory.sqlite"


def init_profile_db():
    """Create the profile table if it does not already exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                updated_at TEXT NOT NULL
            )
        """)


def get_profile(user_id: str) -> dict | None:
    """Retrieve profile by user identity."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        row = conn.execute("""
            SELECT user_id, name, updated_at
            FROM user_profiles
            WHERE user_id = ?
        """, (user_id,)).fetchone()

    return dict(row) if row else None


def update_profile_name(user_id: str, name: str) -> None:
    """Save or update the user's name."""
    name = name.strip()

    if not user_id.strip():
        raise ValueError("user_id cannot be empty.")

    if not name:
        raise ValueError("Name cannot be empty.")

    updated_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO user_profiles (user_id, name, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                name = excluded.name,
                updated_at = excluded.updated_at
        """, (user_id, name, updated_at))


if __name__ == "__main__":
    init_profile_db()
    print(f"Profile database: {DB_PATH}")
    print("Profile database is ready to use.")