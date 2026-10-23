import tempfile
import unittest
from pathlib import Path

import torch

from model import LocalTransformer, ModelConfig
from conversation_store import ConversationStore
from document_store import DocumentStore, chunk_text
from import_wikipedia import clean_wikitext
from github_publisher import teaching_key_matches
from knowledge import KnowledgeBase
from nvidia_client import NvidiaChat
from learning_store import LearningStore, extract_fact
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

    def test_matches_accented_question_to_ascii_training_data(self) -> None:
        knowledge = KnowledgeBase([("ola", "Ola! Como posso ajudar?")])
        self.assertEqual(knowledge.find("olá")[0], "Ola! Como posso ajudar?")


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

    def test_chunks_keep_the_entire_text(self) -> None:
        text = "abcde" * 100
        chunks = list(chunk_text(text, chunk_size=100, overlap=20))
        self.assertTrue(chunks)
        self.assertTrue(chunks[0].startswith("abcde"))
        self.assertTrue(chunks[-1].endswith("abcde"))


class ConversationStoreTests(unittest.TestCase):
    def test_persists_and_groups_chat_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory) / "conversations.db")
            store.add_turn("browser-1", "ola", "Ola! Como posso ajudar?")
            self.assertEqual(store.history("browser-1"), [("ola", "Ola! Como posso ajudar?")])
            self.assertEqual(store.entries("browser-1")[0]["role"], "user")
            store.close()


class LearningStoreTests(unittest.TestCase):
    def test_remembers_a_fact_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LearningStore(Path(directory) / "learning.db")
            self.assertTrue(store.remember("browser-1", "O projeto usa Python."))
            self.assertFalse(store.remember("browser-2", "O projeto usa Python."))
            store.close()

    def test_extracts_teaching_command_without_losing_first_letter(self) -> None:
        self.assertEqual(extract_fact("/ensinar Meu nome e Zeta."), "Meu nome e Zeta.")

    def test_github_sync_requires_the_owner_key(self) -> None:
        self.assertTrue(teaching_key_matches("segredo", "segredo"))
        self.assertFalse(teaching_key_matches("errado", "segredo"))
        self.assertFalse(teaching_key_matches(None, "segredo"))


class NvidiaClientTests(unittest.TestCase):
    def test_builds_portuguese_context_messages(self) -> None:
        client = NvidiaChat()
        messages = client.messages([("ola", "Ola!")], "explique IA", "IA aprende com dados.")
        self.assertEqual(messages[-1], {"role": "user", "content": "explique IA"})
        self.assertIn("IA aprende com dados.", messages[0]["content"])


class WikipediaImportTests(unittest.TestCase):
    def test_cleans_basic_wiki_markup(self) -> None:
        text = "[[Python (linguagem)|Python]] usa {{linguagem}} <ref>fonte</ref>."
        self.assertEqual(clean_wikitext(text), "Python usa .")


if __name__ == "__main__":
    unittest.main()
