from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from chat import load_model, respond
from knowledge import KnowledgeBase
from tokenizer import BPETokenizer

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "conversations.txt"

app = FastAPI(title="MinhaIA V4")
knowledge = KnowledgeBase.from_file(DATA_PATH)
model = None
tokenizer = BPETokenizer.train(DATA_PATH.read_text(encoding="utf-8"))


class ChatMessage(BaseModel):
    message: str


@app.on_event("startup")
def load_runtime() -> None:
    global model, tokenizer
    checkpoint = ROOT / "checkpoints" / "best.pt"
    if checkpoint.exists():
        model, tokenizer = load_model()


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
    if model is None:
        return {
            "answer": "Ainda nao tenho dados suficientes para essa pergunta. Adicione exemplos em data/conversations.txt e execute o treinamento.",
            "source": "limite do modelo",
        }

    answer, _ = respond(model, tokenizer, knowledge, [], question, 120, 0.35)
    return {"answer": answer, "source": "modelo local"}
