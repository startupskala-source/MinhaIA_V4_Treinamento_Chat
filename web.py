from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from chat import load_model, respond
from conversation_store import ConversationStore
from document_store import DocumentStore
from knowledge import KnowledgeBase
from tokenizer import BPETokenizer

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "conversations.txt"

app = FastAPI(title="MinhaIA V4")
knowledge = KnowledgeBase.from_file(DATA_PATH)
model = None
tokenizer = BPETokenizer.train(DATA_PATH.read_text(encoding="utf-8"))
documents = None
conversations = ConversationStore(ROOT / "storage" / "conversations.db")


class ChatMessage(BaseModel):
    message: str
    session_id: str = "anonymous"


@app.on_event("startup")
def load_runtime() -> None:
    global documents, model, tokenizer
    checkpoint = ROOT / "checkpoints" / "best.pt"
    if checkpoint.exists():
        model, tokenizer = load_model()
    database = ROOT / "data" / "knowledge.db"
    if database.exists():
        documents = DocumentStore(database)


@app.on_event("shutdown")
def close_runtime() -> None:
    if documents is not None:
        documents.connection.close()
    conversations.close()


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
    match = knowledge.find(question)
    if match:
        answer, source = match[0], "base de conhecimento"
    else:
        passages = documents.search(question, limit=3) if documents is not None else []
        answer, source = answer_from_documents(question, passages, previous_turns)
    conversations.add_turn(message.session_id, question, answer)
    return {"answer": answer, "source": source, "session_id": message.session_id}


def answer_from_documents(
    question: str, passages: list[dict[str, str | float]], history: list[tuple[str, str]]
) -> tuple[str, str]:
    if passages:
        context = "\n\n".join(passage["content"][:1200] for passage in passages)
        source = Path(str(passages[0]["source"])).name
        if model is None:
            return f"Encontrei este trecho relevante:\n\n{passages[0]['content']}", f"documento local: {source}"
        answer, _ = respond(model, tokenizer, knowledge, history, question, 160, 0.35, context)
        return answer, f"modelo local + documento: {source}"
    if model is None:
        return "Nao encontrei uma fonte local para essa pergunta ainda.", "base local"

    answer, _ = respond(model, tokenizer, knowledge, history, question, 120, 0.35)
    return answer, "modelo local"
