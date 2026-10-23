import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as functional

from model import LocalTransformer, ModelConfig
from tokenizer import BPETokenizer

ROOT = Path(__file__).parent
PRESETS = {
    "tiny": {"context_size": 128, "embedding_size": 128, "attention_heads": 4, "layers": 3},
    "small": {"context_size": 512, "embedding_size": 256, "attention_heads": 8, "layers": 6},
    "base": {"context_size": 1024, "embedding_size": 512, "attention_heads": 8, "layers": 8},
}


def batch_from_memmap(
    data: np.memmap, context_size: int, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - context_size - 1, (batch_size,)).tolist()
    inputs = np.stack([data[index : index + context_size] for index in starts]).astype(np.int64)
    targets = np.stack([data[index + 1 : index + context_size + 1] for index in starts]).astype(np.int64)
    return torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)


@torch.inference_mode()
def evaluate(model: LocalTransformer, data: np.memmap, config: ModelConfig, device: torch.device) -> float:
    model.eval()
    losses = []
    for _ in range(12):
        inputs, targets = batch_from_memmap(data, config.context_size, 4, device)
        logits = model(inputs)
        losses.append(functional.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina um Transformer independente a partir de corpus.bin.")
    parser.add_argument("--corpus", type=Path, default=ROOT / "artifacts" / "corpus.bin")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts" / "tokenizer.json")
    parser.add_argument("--preset", choices=PRESETS, default="tiny")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    args = parser.parse_args()

    if not args.corpus.exists() or not args.tokenizer.exists():
        raise FileNotFoundError("Execute prepare_corpus.py antes de iniciar o treino.")
    tokenizer = BPETokenizer.from_dict(json.loads(args.tokenizer.read_text(encoding="utf-8")))
    tokens = np.memmap(args.corpus, dtype=np.uint16, mode="r")
    config = ModelConfig(vocab_size=tokenizer.vocab_size, **PRESETS[args.preset])
    validation_size = max(config.context_size + 2, int(len(tokens) * 0.02))
    split = len(tokens) - validation_size
    train_data, validation_data = tokens[:split], tokens[split:]
    if min(len(train_data), len(validation_data)) <= config.context_size + 1:
        raise ValueError("Corpus insuficiente para o preset escolhido.")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LocalTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1, fused=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for step in range(1, args.steps + 1):
        inputs, targets = batch_from_memmap(train_data, config.context_size, args.batch_size, device)
        logits = model(inputs)
        loss = functional.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 250 == 0 or step == args.steps:
            validation_loss = evaluate(model, validation_data, config, device)
            checkpoint = {
                "config": config.to_dict(),
                "tokenizer": tokenizer.to_dict(),
                "state_dict": model.state_dict(),
                "validation_loss": validation_loss,
                "step": step,
            }
            torch.save(checkpoint, args.checkpoint_dir / "last.pt")
            if validation_loss < best_loss:
                best_loss = validation_loss
                torch.save(checkpoint, args.checkpoint_dir / "best.pt")
            print(f"etapa {step}/{args.steps} | treino {loss.item():.4f} | validacao {validation_loss:.4f}")

    print(f"Treino encerrado em {device.type}; melhor validacao: {best_loss:.4f}.")


if __name__ == "__main__":
    main()
