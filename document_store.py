import re
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path


class DocumentStore:
    """Disk-backed BM25 search built with SQLite FTS5 and no external service."""

    def __init__(self, database_path: Path):
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database_path)
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self.connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                content,
                source UNINDEXED,
                tokenize = 'unicode61 remove_diacritics 2'
            )
            """
        )
        self.connection.commit()

    def __enter__(self) -> "DocumentStore":
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        self.connection.close()

    def add_chunks(self, chunks: Iterable[tuple[str, str]], batch_size: int = 200) -> int:
        batch: list[tuple[str, str]] = []
        inserted = 0
        for content, source in chunks:
            content = content.strip()
            if len(content) < 40:
                continue
            batch.append((content, source))
            if len(batch) == batch_size:
                self.connection.executemany("INSERT INTO chunks (content, source) VALUES (?, ?)", batch)
                self.connection.commit()
                inserted += len(batch)
                batch.clear()
        if batch:
            self.connection.executemany("INSERT INTO chunks (content, source) VALUES (?, ?)", batch)
            self.connection.commit()
            inserted += len(batch)
        return inserted

    def search(self, question: str, limit: int = 5) -> list[dict[str, str | float]]:
        terms = re.findall(r"[\w]{2,}", question.casefold(), flags=re.UNICODE)
        if not terms:
            return []
        query = " OR ".join(f'"{term}"' for term in terms)
        rows = self.connection.execute(
            """
            SELECT content, source, bm25(chunks) AS score
            FROM chunks
            WHERE chunks MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        return [{"content": row[0], "source": row[1], "score": row[2]} for row in rows]

    def count(self) -> int:
        return self.connection.execute("SELECT count(*) FROM chunks").fetchone()[0]


def chunk_file(path: Path, chunk_size: int = 1400, overlap: int = 220) -> Iterator[str]:
    """Streams a text file into overlapping chunks without loading it all into memory."""
    if overlap >= chunk_size:
        raise ValueError("overlap deve ser menor que chunk_size.")
    carry = ""
    with path.open("r", encoding="utf-8", errors="replace") as file:
        while block := file.read(chunk_size * 8):
            buffer = carry + block
            while len(buffer) >= chunk_size:
                end = buffer.rfind("\n", 0, chunk_size)
                if end < chunk_size // 2:
                    end = chunk_size
                yield buffer[:end]
                buffer = buffer[max(0, end - overlap) :]
            carry = buffer
    if carry.strip():
        yield carry


def source_chunks(paths: Iterable[Path], chunk_size: int, overlap: int) -> Iterator[tuple[str, str]]:
    for path in paths:
        for chunk in chunk_file(path, chunk_size, overlap):
            yield chunk, str(path)
