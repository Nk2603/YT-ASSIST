
import sqlite3
from pathlib import Path
from datetime import datetime

# Project-root/data/chat_history.db
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "chat_history.db"


def get_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT 'New Chat',
                video_url TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            );
        """)


def create_conversation(video_url=None, title="New Chat"):
    now = datetime.now().isoformat(timespec="seconds")

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO conversations
                (title, video_url, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (title, video_url, now, now),
        )
        return cursor.lastrowid


def list_conversations():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, title, video_url, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC, id DESC
        """).fetchall()

        return [dict(row) for row in rows]


def get_conversation(conversation_id):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()

        return dict(row) if row else None


def get_messages(conversation_id):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
        """, (conversation_id,)).fetchall()

        return [dict(row) for row in rows]


def add_message(conversation_id, role, content):
    if role not in ("user", "assistant"):
        raise ValueError("Role must be 'user' or 'assistant'.")

    now = datetime.now().isoformat(timespec="seconds")

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO messages
                (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
        """, (conversation_id, role, str(content), now))

        conn.execute("""
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ?
        """, (now, conversation_id))


def update_title(conversation_id, title):
    title = title.strip()
    if not title:
        raise ValueError("Conversation title cannot be empty.")

    with get_connection() as conn:
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title[:100], conversation_id),
        )


def update_video_url(conversation_id, video_url):
    with get_connection() as conn:
        conn.execute(
            "UPDATE conversations SET video_url = ? WHERE id = ?",
            (video_url, conversation_id),
        )


def delete_conversation(conversation_id):
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
