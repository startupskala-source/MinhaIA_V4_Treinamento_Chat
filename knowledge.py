import math
import re
from collections import Counter
from pathlib import Path


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class KnowledgeBase:
    """A local TF-IDF retriever built only from the project's conversation file."""

    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs
        document_frequency = Counter()
        self.documents = []
        for question, _ in pairs:
            counts = Counter(words(question))
            self.documents.append(counts)
            document_frequency.update(counts.keys())
        total_documents = len(pairs)
        self.idf = {
            term: math.log((1 + total_documents) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }

    @classmethod
    def from_file(cls, path: Path) -> "KnowledgeBase":
        pairs: list[tuple[str, str]] = []
        question: str | None = None
        answer_lines: list[str] = []
        mode = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            if line == "### USUARIO":
                mode = "question"
                question = ""
                answer_lines = []
            elif line == "### ASSISTENTE":
                mode = "answer"
            elif line == "### FIM":
                if question and answer_lines:
                    pairs.append((question.strip(), "\n".join(answer_lines).strip()))
                mode = ""
            elif mode == "question":
                question = f"{question}\n{line}" if question else line
            elif mode == "answer":
                answer_lines.append(line)
        return cls(pairs)

    def find(self, question: str) -> tuple[str, float] | None:
        query = Counter(words(question))
        if not query:
            return None
        query_norm = math.sqrt(sum((count * self.idf.get(term, 0)) ** 2 for term, count in query.items()))
        best_answer = ""
        best_score = 0.0
        for document, (_, answer) in zip(self.documents, self.pairs):
            dot = sum(
                query_count * self.idf.get(term, 0) * document.get(term, 0) * self.idf.get(term, 0)
                for term, query_count in query.items()
            )
            document_norm = math.sqrt(
                sum((count * self.idf.get(term, 0)) ** 2 for term, count in document.items())
            )
            score = dot / (query_norm * document_norm) if query_norm and document_norm else 0.0
            if score > best_score:
                best_answer, best_score = answer, score
        return (best_answer, best_score) if best_score >= 0.32 else None
