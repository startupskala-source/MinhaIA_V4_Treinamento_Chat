import sqlite3
from pathlib import Path


class UsageStore:
    def __init__(self, database_path: Path):
        database_path.parent.mkdir(exist_ok=True)
        self.connection = sqlite3.connect(database_path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS usage (user_id TEXT PRIMARY KEY, tokens INTEGER NOT NULL DEFAULT 0)"
        )
        self.connection.commit()

    def __enter__(self) -> "UsageStore":
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        self.connection.close()

    def used(self, user_id: str) -> int:
        row = self.connection.execute("SELECT tokens FROM usage WHERE user_id = ?", (user_id,)).fetchone()
        return 0 if row is None else row[0]

    def require_available(self, user_id: str, limit: int, requested: int) -> None:
        remaining = limit - self.used(user_id)
        if requested > remaining:
            raise ValueError(f"Cota insuficiente: restam {remaining} tokens de {limit}.")

    def consume(self, user_id: str, amount: int) -> int:
        self.connection.execute(
            """
            INSERT INTO usage (user_id, tokens) VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET tokens = tokens + excluded.tokens
            """,
            (user_id, amount),
        )
        self.connection.commit()
        return self.used(user_id)
