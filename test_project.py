import tempfile
import unittest
from pathlib import Path

import torch

from model import LocalTransformer, ModelConfig
from document_store import DocumentStore
from knowledge import KnowledgeBase
from storage import UsageStore
from tokenizer import BPETokenizer


class TokenizerTests(unittest.TestCase):
    def test_bpe_preserves_utf8_text(self) -> None:
        text = "Olá, inteligência artificial!"
        tokenizer = BPETokenizer.train(text * 4, vocab_size=300)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_bpe_can_be_restored_from_serialized_data(self) -> None:
        tokenizer = BPETokenizer.train("dados locais " * 20, vocab_size=280)
        restored = BPETokenizer.from_dict(tokenizer.to_dict())
        self.assertEqual(restored.decode(restored.encode("dados locais")), "dados locais")


class ModelTests(unittest.TestCase):
    def test_transformer_produces_logits_for_each_token(self) -> None:
        config = ModelConfig(vocab_size=12, context_size=16, embedding_size=12, attention_heads=2, layers=1)
        model = LocalTransformer(config)
        logits = model(torch.tensor([[1, 2, 3, 4]]))
        self.assertEqual(tuple(logits.shape), (1, 4, 12))


class UsageTests(unittest.TestCase):
    def test_usage_is_counted_and_limited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with UsageStore(Path(directory) / "usage.db") as usage:
                usage.require_available("ana", 10, 6)
                self.assertEqual(usage.consume("ana", 6), 6)
                with self.assertRaises(ValueError):
                    usage.require_available("ana", 10, 5)


class KnowledgeTests(unittest.TestCase):
    def test_retrieves_related_question(self) -> None:
        knowledge = KnowledgeBase(
            [("o que e machine learning", "Machine learning aprende com exemplos.")]
        )
        self.assertEqual(knowledge.find("me explique machine learning")[0], "Machine learning aprende com exemplos.")


class DocumentStoreTests(unittest.TestCase):
    def test_bm25_returns_relevant_disk_backed_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with DocumentStore(Path(directory) / "knowledge.db") as store:
                store.add_chunks(
                    [
                        ("Machine learning aprende padroes a partir de exemplos de treinamento.", "ia.txt"),
                        ("Uma planta precisa de luz e agua para crescer.", "plantas.txt"),
                    ]
                )
                results = store.search("como machine learning aprende")
                self.assertEqual(results[0]["source"], "ia.txt")


if __name__ == "__main__":
    unittest.main()
