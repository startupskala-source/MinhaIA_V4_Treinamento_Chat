import argparse
import bz2
import re
import urllib.request
import xml.etree.ElementTree as element_tree
from collections.abc import Iterator
from pathlib import Path

from document_store import DocumentStore, chunk_text

ROOT = Path(__file__).parent


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def clean_wikitext(text: str) -> str:
    text = re.sub(r"<ref[^>/]*?>.*?</ref>|<ref[^>]*/>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    for _ in range(8):
        reduced = re.sub(r"\{\{[^{}]*\}\}", " ", text)
        if reduced == text:
            break
        text = reduced
    text = re.sub(r"\[\[([^\]|]*\|)?([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"==+.*?==+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def wikipedia_articles(url: str, maximum_articles: int | None) -> Iterator[tuple[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "MinhaIA-V4/1.0 (local educational indexer)"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with bz2.BZ2File(response) as archive:
            articles = 0
            for _, page in element_tree.iterparse(archive, events=("end",)):
                if local_name(page.tag) != "page":
                    continue
                title = ""
                namespace = ""
                content = ""
                for node in page.iter():
                    name = local_name(node.tag)
                    if name == "title":
                        title = node.text or ""
                    elif name == "ns":
                        namespace = node.text or ""
                    elif name == "text":
                        content = node.text or ""
                page.clear()
                if namespace != "0" or not content:
                    continue
                cleaned = clean_wikitext(content)
                if len(cleaned) < 300:
                    continue
                yield title, cleaned
                articles += 1
                if maximum_articles is not None and articles >= maximum_articles:
                    return


def main() -> None:
    parser = argparse.ArgumentParser(description="Importa Wikipedia para a base local da MinhaIA.")
    parser.add_argument("--language", default="pt", help="Código Wikimedia, como pt ou en.")
    parser.add_argument("--url", help="URL alternativa de dump .xml.bz2.")
    parser.add_argument("--database", type=Path, default=ROOT / "data" / "knowledge.db")
    parser.add_argument("--max-articles", type=int, help="Limite para teste; omita para importar todo o dump.")
    parser.add_argument("--chunk-size", type=int, default=1400)
    parser.add_argument("--overlap", type=int, default=220)
    args = parser.parse_args()

    url = args.url or f"https://dumps.wikimedia.org/{args.language}wiki/latest/{args.language}wiki-latest-pages-articles-multistream.xml.bz2"

    def chunks() -> Iterator[tuple[str, str]]:
        for index, (title, article) in enumerate(wikipedia_articles(url, args.max_articles), start=1):
            if index % 500 == 0:
                print(f"Processados {index} artigos...")
            source = f"wikipedia:{args.language}:{title}"
            for chunk in chunk_text(article, args.chunk_size, args.overlap):
                yield chunk, source

    with DocumentStore(args.database) as store:
        inserted = store.add_chunks(chunks())
        print(f"Importação concluída: {inserted} trechos. Total: {store.count()}.")


if __name__ == "__main__":
    main()
