from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int
    context_size: int = 128
    embedding_size: int = 128
    attention_heads: int = 4
    layers: int = 3
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class LocalTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.position_embedding = nn.Embedding(config.context_size, config.embedding_size)
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_size,
            nhead=config.attention_heads,
            dim_feedforward=config.embedding_size * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.norm = nn.LayerNorm(config.embedding_size)
        self.output = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, length = tokens.shape
        if length > self.config.context_size:
            raise ValueError("A entrada excede o contexto configurado.")
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = torch.triu(
            torch.ones(length, length, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        return self.output(self.norm(self.transformer(hidden, mask=mask)))

    @torch.inference_mode()
    def generate(
        self,
        prompt: list[int],
        max_new_tokens: int,
        temperature: float,
        stop_sequence: list[int],
    ) -> list[int]:
        if temperature <= 0:
            raise ValueError("temperature deve ser maior que zero.")
        generated = list(prompt)
        device = next(self.parameters()).device
        for _ in range(max_new_tokens):
            context = generated[-self.config.context_size :]
            logits = self(torch.tensor([context], dtype=torch.long, device=device))[0, -1]
            probabilities = torch.softmax(logits / temperature, dim=-1)
            generated.append(torch.multinomial(probabilities, 1).item())
            if len(generated) >= len(stop_sequence) and generated[-len(stop_sequence) :] == stop_sequence:
                break
        return generated
