
from pathlib import Path
import sqlite3

DB_PATH = Path("checkpoints.sqlite")

if not DB_PATH.is_file():
    raise FileNotFoundError(
        f"Database tidak ditemukan: {DB_PATH.resolve()}"
    )

print(f"Database: {DB_PATH.resolve()}")
print(f"Ukuran  : {DB_PATH.stat().st_size:,} bytes")

with sqlite3.connect(DB_PATH) as conn:
    tables = conn.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
    """).fetchall()


    if not tables:
        print("\nBelum ada tabel di database.")
    else:
        print("\n=== Tabel SQLite ===")

        for (table_name,) in tables:
            columns = conn.execute(
                f'PRAGMA table_info("{table_name}")'
            ).fetchall()

            row_count = conn.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]

            column_names = [
                column[1] for column in columns
            ]

            print(f"\nTabel       : {table_name}")
            print(f"Jumlah row  : {row_count}")
            print(f"Kolom       : {', '.join(column_names)}")

    print("\n=== Checkpoint per thread ===")
    
    rows = conn.execute("""
        SELECT
            thread_id,
            checkpoint_ns,
            COUNT(*) AS checkpoint_count,
            MAX(checkpoint_id) AS latest_checkpoint_id
        FROM checkpoints
        GROUP BY thread_id, checkpoint_ns
        ORDER BY thread_id, checkpoint_ns
    """).fetchall()

    for thread_id, namespace, count, latest_id in rows:
        print(f"\nThread ID          : {thread_id}")
        print(f"Namespace          : {namespace!r}")
        print(f"Jumlah checkpoint  : {count}")
        print(f"Checkpoint ID akhir: {latest_id}")

    
    print("\n=== Metadata checkpoint terbaru ===")

    latest_rows = conn.execute("""
        SELECT c.thread_id, c.checkpoint_ns, c.checkpoint_id,
               c.parent_checkpoint_id, c.metadata
        FROM checkpoints AS c
        INNER JOIN (
            SELECT thread_id, checkpoint_ns,
                   MAX(rowid) AS latest_rowid
            FROM checkpoints
            GROUP BY thread_id, checkpoint_ns
        ) AS latest
        ON c.rowid = latest.latest_rowid
        ORDER BY c.thread_id
    """).fetchall()

    for thread_id, namespace, checkpoint_id, parent_id, metadata in latest_rows:
        print(f"\nThread ID       : {thread_id}")
        print(f"Namespace       : {namespace!r}")
        print(f"Checkpoint ID   : {checkpoint_id}")
        print(f"Parent ID       : {parent_id}")
        print(f"Metadata mentah : {metadata}")
