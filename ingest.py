import argparse
from pathlib import Path

from document_store import DocumentStore, source_chunks

ROOT = Path(__file__).parent
TEXT_SUFFIXES = {".txt", ".md", ".csv", ".jsonl"}


def input_files(location: Path) -> list[Path]:
    if location.is_file():
        return [location]
    return [path for path in location.rglob("*") if path.suffix.lower() in TEXT_SUFFIXES]


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa documentos locais no banco BM25 da MinhaIA.")
    parser.add_argument("path", type=Path, help="Arquivo ou pasta contendo documentos de texto.")
    parser.add_argument("--database", type=Path, default=ROOT / "data" / "knowledge.db")
    parser.add_argument("--chunk-size", type=int, default=1400)
    parser.add_argument("--overlap", type=int, default=220)
    args = parser.parse_args()

    files = input_files(args.path)
    if not files:
        raise FileNotFoundError("Nenhum arquivo .txt, .md, .csv ou .jsonl foi encontrado.")
    with DocumentStore(args.database) as store:
        added = store.add_chunks(source_chunks(files, args.chunk_size, args.overlap))
        print(f"{added} trechos adicionados. Total indexado: {store.count()}.")


if __name__ == "__main__":
    main()
