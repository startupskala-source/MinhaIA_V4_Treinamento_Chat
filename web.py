import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from chat import load_model, respond
from conversation_store import ConversationStore
from document_store import DocumentStore
from github_publisher import GitHubPublisher, teaching_key_matches
from knowledge import KnowledgeBase
from learning_store import LearningStore, extract_fact
from nvidia_client import NvidiaChat, NvidiaError
from tokenizer import BPETokenizer

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "conversations.txt"

app = FastAPI(title="MinhaIA V4")
knowledge = KnowledgeBase.from_file(DATA_PATH)
model = None
tokenizer = BPETokenizer.train(DATA_PATH.read_text(encoding="utf-8"))
documents = None
conversations = ConversationStore(ROOT / "storage" / "conversations.db")
learning = LearningStore(ROOT / "storage" / "learning.db")
publisher = GitHubPublisher()
nvidia = NvidiaChat()


class ChatMessage(BaseModel):
    message: str
    session_id: str = "anonymous"
    learning_key: str | None = None


@app.on_event("startup")
def load_runtime() -> None:
    global documents, model, tokenizer
    checkpoint = ROOT / "checkpoints" / "best.pt"
    if checkpoint.exists():
        model, tokenizer = load_model()
    documents = DocumentStore(ROOT / "data" / "knowledge.db")
    load_committed_memories()


def load_committed_memories() -> None:
    for path in (ROOT / "data" / "learned").glob("*.jsonl"):
        source = f"learned:{path.name}"
        if documents.has_source(source):
            continue
        facts = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                fact = json.loads(line).get("fact", "")
            except json.JSONDecodeError:
                continue
            if fact:
                facts.append((fact, source))
        documents.add_chunks(facts)


@app.on_event("shutdown")
def close_runtime() -> None:
    if documents is not None:
        documents.connection.close()
    conversations.close()
    learning.close()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/history/{session_id}")
def history(session_id: str) -> dict[str, list[dict[str, str]]]:
    return {"messages": conversations.entries(session_id)}


@app.post("/api/chat")
def chat(message: ChatMessage) -> dict[str, str]:
    question = message.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Envie uma pergunta antes de continuar.")

    previous_turns = conversations.history(message.session_id)
    fact = extract_fact(question)
    if fact is not None:
        answer, source = teach(message.session_id, fact, message.learning_key)
    else:
        passages = documents.search(question, limit=5) if documents is not None else []
        learned_passages = [
            passage
            for passage in passages
            if str(passage["source"]).startswith(("memory:", "learned:"))
        ]
        match = knowledge.find(question)
        if learned_passages:
            remembered_fact = str(learned_passages[0]["content"])
            answer = f"Eu me lembro: {remembered_fact}"
            source = f"memoria ensinada: {learned_passages[0]['source']}"
        elif match:
            answer, source = match[0], "base de conhecimento"
        else:
            answer, source = answer_from_documents(question, passages, previous_turns)
    conversations.add_turn(message.session_id, question, answer)
    return {"answer": answer, "source": source, "session_id": message.session_id}


def teach(session_id: str, fact: str, learning_key: str | None) -> tuple[str, str]:
    try:
        is_new = learning.remember(session_id, fact)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if not is_new:
        return "Eu ja tinha essa informacao na minha memoria.", "memoria local"
    documents.add_chunks([(fact, f"memory:{session_id}")])
    owner_key = os.getenv("TEACHING_KEY")
    published_url = None
    if publisher.enabled and teaching_key_matches(learning_key, owner_key):
        try:
            published_url = publisher.publish_fact(fact, session_id)
        except Exception:
            published_url = None
    if published_url:
        return "Aprendi e sincronizei esta memoria com o GitHub.", "memoria + GitHub"
    if publisher.enabled:
        return "Aprendi esta informacao localmente. A sincronizacao com GitHub exige a chave de proprietario.", "memoria local"
    return "Aprendi esta informacao na memoria local.", "memoria local"


def answer_from_documents(
    question: str, passages: list[dict[str, str | float]], history: list[tuple[str, str]]
) -> tuple[str, str]:
    if passages:
        context = "\n\n".join(passage["content"][:1200] for passage in passages)
        source = Path(str(passages[0]["source"])).name
        if nvidia.enabled:
            try:
                return nvidia.complete(history, question, context), f"NVIDIA NIM + documento: {source}"
            except NvidiaError:
                pass
        if model is None:
            return f"Encontrei este trecho relevante:\n\n{passages[0]['content']}", f"documento local: {source}"
        answer, _ = respond(model, tokenizer, knowledge, history, question, 160, 0.35, context)
        return answer, f"modelo local + documento: {source}"
    if nvidia.enabled:
        try:
            return nvidia.complete(history, question), "NVIDIA NIM"
        except NvidiaError:
            pass
    if model is None:
        return "Nao encontrei uma fonte local para essa pergunta ainda.", "base local"

    answer, _ = respond(model, tokenizer, knowledge, history, question, 120, 0.35)
    return answer, "modelo local"
