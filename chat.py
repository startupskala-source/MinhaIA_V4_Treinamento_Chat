import argparse
import json
from pathlib import Path

import torch

from model import LocalTransformer, ModelConfig
from knowledge import KnowledgeBase
from storage import UsageStore
from tokenizer import BPETokenizer

ROOT = Path(__file__).parent


def load_model() -> tuple[LocalTransformer, BPETokenizer]:
    checkpoint_path = ROOT / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("Modelo ausente. Execute: python train.py")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = LocalTransformer(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, BPETokenizer(**checkpoint["tokenizer"])


def build_prompt(history: list[tuple[str, str]], message: str) -> str:
    recent_history = history[-3:]
    prompt = "".join(
        f"### USUARIO\n{old_message}\n### ASSISTENTE\n{old_response}\n### FIM\n\n"
        for old_message, old_response in recent_history
    )
    return prompt + f"### USUARIO\n{message}\n### ASSISTENTE\n"


def respond(
    model: LocalTransformer,
    tokenizer: BPETokenizer,
    knowledge: KnowledgeBase,
    history: list[tuple[str, str]],
    message: str,
    max_output_tokens: int,
    temperature: float,
) -> tuple[str, int]:
    prompt = build_prompt(history, message)
    prompt_tokens = tokenizer.encode(prompt)
    match = knowledge.find(message)
    if match:
        response_tokens = tokenizer.encode(match[0])[:max_output_tokens]
        return tokenizer.decode(response_tokens), len(prompt_tokens) + len(response_tokens)
    generated = model.generate(
        prompt_tokens,
        max_output_tokens,
        temperature,
        tokenizer.encode("\n### FIM"),
    )
    response_tokens = generated[len(prompt_tokens) :]
    end_marker = tokenizer.encode("\n### FIM")
    if response_tokens[-len(end_marker) :] == end_marker:
        response_tokens = response_tokens[: -len(end_marker)]
    return tokenizer.decode(response_tokens).strip(), len(prompt_tokens) + len(response_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Converse com a IA local.")
    parser.add_argument("--user", required=True)
    parser.add_argument("--message", help="Mensagem única; sem esta opção abre o chat interativo.")
    parser.add_argument("--max-tokens", type=int)
    args = parser.parse_args()

    users = json.loads((ROOT / "data" / "users.json").read_text(encoding="utf-8"))
    if args.user not in users:
        raise ValueError(f"Usuário desconhecido: {args.user}")
    profile = users[args.user]
    max_output = args.max_tokens or profile["max_output_tokens"]
    if not 1 <= max_output <= profile["max_output_tokens"]:
        raise ValueError("max-tokens está fora do limite configurado para este usuário.")

    model, tokenizer = load_model()
    knowledge = KnowledgeBase.from_file(ROOT / "data" / "conversations.txt")
    history: list[tuple[str, str]] = []
    messages = [args.message] if args.message else None
    while True:
        message = messages.pop(0) if messages else input("Você: ").strip()
        if not message or message.lower() in {"sair", "exit", "quit"}:
            break

        with UsageStore(ROOT / "storage" / "usage.db") as usage:
            estimated = len(tokenizer.encode(build_prompt(history, message))) + max_output
            usage.require_available(args.user, profile["token_limit"], estimated)
            response, consumed = respond(
                model, tokenizer, knowledge, history, message, max_output, profile["temperature"]
            )
            total = usage.consume(args.user, consumed)
        print(f"IA: {response}")
        print(f"Uso: {total}/{profile['token_limit']} tokens")
        history.append((message, response))
        if args.message:
            break


if __name__ == "__main__":
    main()
