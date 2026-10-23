import argparse
import json
from pathlib import Path

import numpy as np

from ingest import input_files
from tokenizer import BPETokenizer

ROOT = Path(__file__).parent


def training_sample(files: list[Path], maximum_characters: int) -> str:
    parts: list[str] = []
    remaining = maximum_characters
    for path in files:
        if remaining <= 0:
            break
        with path.open("r", encoding="utf-8", errors="replace") as file:
            part = file.read(remaining)
        parts.append(part)
        remaining -= len(part)
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara um corpus grande para treino sem mantê-lo inteiro em RAM.")
    parser.add_argument("path", type=Path, help="Arquivo ou diretório com textos de treinamento.")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "corpus.bin")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts" / "tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--sample-chars", type=int, default=150_000)
    parser.add_argument("--block-chars", type=int, default=65_536)
    args = parser.parse_args()

    if not 256 <= args.vocab_size <= 65_535:
        raise ValueError("vocab-size deve estar entre 256 e 65535.")
    files = input_files(args.path)
    if not files:
        raise FileNotFoundError("Nenhum texto para preparar.")
    sample = training_sample(files, args.sample_chars)
    if not sample:
        raise ValueError("Os arquivos de treinamento estão vazios.")
    tokenizer = BPETokenizer.train(sample, args.vocab_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tokenizer.parent.mkdir(parents=True, exist_ok=True)
    token_count = 0
    with args.output.open("wb") as destination:
        for path in files:
            with path.open("r", encoding="utf-8", errors="replace") as source:
                while block := source.read(args.block_chars):
                    encoded = np.asarray(tokenizer.encode(block), dtype=np.uint16)
                    encoded.tofile(destination)
                    token_count += len(encoded)

    args.tokenizer.write_text(json.dumps(tokenizer.to_dict()), encoding="utf-8")
    metadata = {
        "token_count": token_count,
        "vocab_size": tokenizer.vocab_size,
        "dtype": "uint16",
        "source_files": [str(path) for path in files],
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Corpus preparado: {token_count} tokens em {args.output}.")


if __name__ == "__main__":
    main()
