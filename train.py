import argparse
from pathlib import Path

import torch
from torch.nn import functional as functional

from model import LocalTransformer, ModelConfig
from tokenizer import BPETokenizer

ROOT = Path(__file__).parent


def make_batch(data: torch.Tensor, context_size: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - context_size - 1, (batch_size,))
    inputs = torch.stack([data[index : index + context_size] for index in starts]).to(device)
    targets = torch.stack([data[index + 1 : index + context_size + 1] for index in starts]).to(device)
    return inputs, targets


@torch.inference_mode()
def evaluate(model: LocalTransformer, data: torch.Tensor, config: ModelConfig, device: torch.device) -> float:
    model.eval()
    losses = []
    for _ in range(10):
        inputs, targets = make_batch(data, config.context_size, 8, device)
        logits = model(inputs)
        losses.append(functional.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o Transformer local do zero.")
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=512)
    args = parser.parse_args()

    torch.manual_seed(42)
    source = (ROOT / "data" / "conversations.txt").read_text(encoding="utf-8")
    tokenizer = BPETokenizer.train(source, args.vocab_size)
    tokens = torch.tensor(tokenizer.encode(source), dtype=torch.long)
    split = int(len(tokens) * 0.9)
    train_data, validation_data = tokens[:split], tokens[split:]
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    if min(len(train_data), len(validation_data)) <= config.context_size + 1:
        raise ValueError("São necessários mais dados para treinar e validar com este contexto.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LocalTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    best_loss = float("inf")
    checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    for step in range(1, args.steps + 1):
        inputs, targets = make_batch(train_data, config.context_size, args.batch_size, device)
        logits = model(inputs)
        loss = functional.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 100 == 0 or step == args.steps:
            validation_loss = evaluate(model, validation_data, config, device)
            print(f"etapa {step}/{args.steps} | treino {loss.item():.4f} | validação {validation_loss:.4f}")
            if validation_loss < best_loss:
                best_loss = validation_loss
                torch.save(
                    {
                        "config": config.to_dict(),
                        "tokenizer": tokenizer.to_dict(),
                        "state_dict": model.state_dict(),
                        "validation_loss": validation_loss,
                    },
                    checkpoint_dir / "best.pt",
                )

    print(f"Modelo salvo em {checkpoint_dir / 'best.pt'} usando {device.type}.")


if __name__ == "__main__":
    main()
