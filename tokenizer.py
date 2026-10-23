from collections import Counter


class BPETokenizer:
    """Byte Pair Encoding trained entirely from the project's own corpus."""

    def __init__(self, merges: list[tuple[int, int]]):
        self.merges = [tuple(pair) for pair in merges]
        self.vocabulary = {index: bytes([index]) for index in range(256)}
        for offset, (left, right) in enumerate(self.merges):
            self.vocabulary[256 + offset] = self.vocabulary[left] + self.vocabulary[right]

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    @classmethod
    def train(cls, text: str, vocab_size: int = 512) -> "BPETokenizer":
        tokens = list(text.encode("utf-8"))
        merges: list[tuple[int, int]] = []

        while 256 + len(merges) < vocab_size and len(tokens) > 1:
            pairs = Counter(zip(tokens, tokens[1:]))
            if not pairs:
                break
            pair, occurrences = pairs.most_common(1)[0]
            if occurrences < 2:
                break
            new_id = 256 + len(merges)
            tokens = cls._merge_pair(tokens, pair, new_id)
            merges.append(pair)

        return cls(merges)

    @staticmethod
    def _merge_pair(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        merged: list[int] = []
        index = 0
        while index < len(tokens):
            if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
                merged.append(new_id)
                index += 2
            else:
                merged.append(tokens[index])
                index += 1
        return merged

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        for offset, pair in enumerate(self.merges):
            tokens = self._merge_pair(tokens, pair, 256 + offset)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        raw = b"".join(self.vocabulary[token] for token in tokens)
        return raw.decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {"merges": [list(pair) for pair in self.merges]}
