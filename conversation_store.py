import sqlite3
import threading
from pathlib import Path


class ConversationStore:
    """Persistent local conversation memory, isolated by browser session identifier."""

    def __init__(self, database_path: Path):
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database_path, check_same_thread=False)
        self.lock = threading.Lock()
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS messages_session_id ON messages(session_id, id)"
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def add_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        with self.lock:
            self.connection.executemany(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                [(session_id, "user", user_message), (session_id, "assistant", assistant_message)],
            )
            self.connection.execute(
                """
                DELETE FROM messages
                WHERE session_id = ? AND id NOT IN (
                    SELECT id FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT 60
                )
                """,
                (session_id, session_id),
            )
            self.connection.commit()

    def history(self, session_id: str, turns: int = 6) -> list[tuple[str, str]]:
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT role, content FROM messages
                WHERE session_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (session_id, turns * 2),
            ).fetchall()
        rows.reverse()
        result: list[tuple[str, str]] = []
        pending_user = None
        for role, content in rows:
            if role == "user":
                pending_user = content
            elif pending_user is not None:
                result.append((pending_user, content))
                pending_user = None
        return result

    def entries(self, session_id: str, limit: int = 30) -> list[dict[str, str]]:
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT role, content FROM messages
                WHERE session_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [{"role": role, "content": content} for role, content in reversed(rows)]
