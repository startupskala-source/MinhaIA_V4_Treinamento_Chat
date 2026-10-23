import sqlite3
import threading
from pathlib import Path

TEACH_PREFIX = "/ensinar "


def extract_fact(message: str) -> str | None:
    if not message.casefold().startswith(TEACH_PREFIX):
        return None
    return message[len(TEACH_PREFIX) :].strip()


class LearningStore:
    """Stores explicit user-taught facts separately from ordinary chat history."""

    def __init__(self, database_path: Path):
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database_path, check_same_thread=False)
        self.lock = threading.Lock()
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS learned_facts (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def remember(self, session_id: str, content: str) -> bool:
        content = content.strip()
        if len(content) < 4:
            raise ValueError("O conhecimento precisa ter pelo menos quatro caracteres.")
        with self.lock:
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO learned_facts (session_id, content) VALUES (?, ?)",
                (session_id, content),
            )
            self.connection.commit()
        return cursor.rowcount == 1
