from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from chat import load_model, respond
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


class ChatMessage(BaseModel):
    message: str


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


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


@app.post("/api/chat")
def chat(message: ChatMessage) -> dict[str, str]:
    question = message.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Envie uma pergunta antes de continuar.")

    match = knowledge.find(question)
    if match:
        return {"answer": match[0], "source": "base de conhecimento"}
    passages = documents.search(question, limit=3) if documents is not None else []
    if passages:
        context = "\n\n".join(passage["content"][:1200] for passage in passages)
        source = Path(str(passages[0]["source"])).name
        if model is None:
            return {
                "answer": f"Encontrei este trecho relevante:\n\n{passages[0]['content']}",
                "source": f"documento local: {source}",
            }
        answer, _ = respond(model, tokenizer, knowledge, [], question, 160, 0.35, context)
        return {"answer": answer, "source": f"modelo local + documento: {source}"}
    if model is None:
        return {
            "answer": "Nao encontrei uma fonte local para essa pergunta ainda.",
            "source": "base local",
        }

    answer, _ = respond(model, tokenizer, knowledge, [], question, 120, 0.35)
    return {"answer": answer, "source": "modelo local"}
