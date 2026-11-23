
from __future__ import annotations

import json
import os
import re
import tempfile

from collections import Counter
from pathlib import Path
from typing import Iterable, Optional


class ByteLevelBPETokenizer:
    """
    Educational production-oriented Byte-Level BPE tokenizer.

    Features:
        - UTF-8 byte-level tokenization
        - BPE training
        - Frequency-based pair merging
        - Rank-based encoding
        - Token ID conversion
        - Special tokens
        - Decoding
        - JSON serialization
        - Tokenizer validation

    Note:
        This is a transparent reference implementation, not a
        performance-equivalent replacement for tiktoken or Hugging Face.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        vocab_size: int = 1000,
        min_frequency: int = 2,
        special_tokens: Optional[list[str]] = None,
    ) -> None:

        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256.")

        if min_frequency < 1:
            raise ValueError("min_frequency must be >= 1.")

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

        self.special_tokens = special_tokens or [
            "[PAD]",
            "[UNK]",
            "[BOS]",
            "[EOS]",
        ]

        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("Special tokens must be unique.")

        if not all(
            isinstance(token, str) and token
            for token in self.special_tokens
        ):
            raise ValueError("Special tokens must be non-empty strings.")

        # Byte sequence -> token ID
        self.token_to_id: dict[bytes, int] = {}

        # Token ID -> byte sequence
        self.id_to_token: dict[int, bytes] = {}

        # (left bytes, right bytes) -> merge rank
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {}

        # Special token string -> ID
        self.special_to_id: dict[str, int] = {}

        # Special token ID -> string
        self.id_to_special: dict[int, str] = {}

        self.is_trained = False

        # Preserve whitespace and non-whitespace chunks.
        # This is deliberately simpler than GPT-2's regex.
        self.pretoken_pattern = re.compile(r"\s+|[^\s]+")

        self._initialize_base_vocabulary()

    # ---------------------------------------------------------
    # 1. INITIALIZATION
    # ---------------------------------------------------------

    def _initialize_base_vocabulary(self) -> None:
        """
        Initialize the 256 possible UTF-8 byte values.

        Each byte initially represents one token.
        """

        for byte_value in range(256):

            token = bytes([byte_value])

            self.token_to_id[token] = byte_value
            self.id_to_token[byte_value] = token

    # ---------------------------------------------------------
    # 2. PRE-TOKENIZATION
    # ---------------------------------------------------------

    def _pre_tokenize(self, text: str) -> list[bytes]:
        """
        Split text into whitespace and non-whitespace chunks.

        Convert each chunk into UTF-8 bytes.

        Whitespace is preserved, not discarded.
        """

        if not isinstance(text, str):
            raise TypeError("Input must be a string.")

        chunks = self.pretoken_pattern.findall(text)

        return [
            chunk.encode("utf-8")
            for chunk in chunks
        ]

    # ---------------------------------------------------------
    # 3. INITIALIZE TRAINING VOCABULARY
    # ---------------------------------------------------------

    def _initialize_training_vocab(
        self,
        word_counts: Counter[bytes],
    ) -> dict[tuple[bytes, ...], int]:
        """
        Convert byte sequences into sequences of individual bytes.
        """

        vocab = {}

        for chunk, frequency in word_counts.items():

            symbols = tuple(
                bytes([byte_value])
                for byte_value in chunk
            )

            vocab[symbols] = frequency

        return vocab

    # ---------------------------------------------------------
    # 4. PAIR FREQUENCY
    # ---------------------------------------------------------

    def _get_pair_frequencies(
        self,
        vocab: dict[tuple[bytes, ...], int],
    ) -> Counter[tuple[bytes, bytes]]:
        """
        Calculate frequency of adjacent byte-token pairs.
        """

        pair_counts = Counter()

        for symbols, frequency in vocab.items():

            for i in range(len(symbols) - 1):

                pair = (
                    symbols[i],
                    symbols[i + 1],
                )

                pair_counts[pair] += frequency

        return pair_counts

    # ---------------------------------------------------------
    # 5. MERGE PAIR
    # ---------------------------------------------------------

    def _merge_pair(
        self,
        pair: tuple[bytes, bytes],
        vocab: dict[tuple[bytes, ...], int],
    ) -> dict[tuple[bytes, ...], int]:
        """
        Merge all non-overlapping occurrences of a selected pair.
        """

        first, second = pair

        merged_token = first + second

        new_vocab = Counter()

        for symbols, frequency in vocab.items():

            new_symbols = []

            i = 0

            while i < len(symbols):

                if (
                    i + 1 < len(symbols)
                    and symbols[i] == first
                    and symbols[i + 1] == second
                ):

                    new_symbols.append(merged_token)
                    i += 2

                else:

                    new_symbols.append(symbols[i])
                    i += 1

            new_vocab[tuple(new_symbols)] += frequency

        return dict(new_vocab)

    # ---------------------------------------------------------
    # 6. TRAIN
    # ---------------------------------------------------------

    def train(
        self,
        texts: Iterable[str],
        max_merges: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """
        Train Byte-Level BPE on an iterable of text strings.

        vocab_size includes:
            - 256 base byte tokens
            - learned merge tokensif 
            - special tokens

        max_merges optionally limits the number of learned merges.
        """

        if isinstance(texts, str):
            raise TypeError(
                "Pass an iterable of text samples, not one string."
            )

        if self.is_trained:
            raise RuntimeError(
                "Tokenizer is already trained. Create a new instance."
            )

        if max_merges is not None and max_merges < 0:
            raise ValueError("max_merges cannot be negative.")

        # Count pre-tokenized chunks.
        chunk_counts = Counter()

        for text in texts:

            for chunk in self._pre_tokenize(text):

                if chunk:
                    chunk_counts[chunk] += 1

        if not chunk_counts:
            raise ValueError("Training corpus is empty.")

        # Convert chunks into sequences of individual bytes.
        vocab = self._initialize_training_vocab(chunk_counts)

        special_count = len(self.special_tokens)

        # The initial vocabulary already has 256 byte tokens.
        available_merges = max(
            0,
            self.vocab_size - 256 - special_count
        )

        if max_merges is not None:
            available_merges = min(
                available_merges,
                max_merges
            )

        # Learn merge rules.
        for step in range(available_merges):

            pair_counts = self._get_pair_frequencies(vocab)

            if not pair_counts:
                break

            # Deterministic tie-breaking:
            # highest frequency, then lexicographically smallest pair.
            best_pair, best_frequency = min(
                pair_counts.items(),
                key=lambda item: (
                    -item[1],
                    item[0][0],
                    item[0][1],
                )
            )

            if best_frequency < self.min_frequency:
                break

            merged_token = best_pair[0] + best_pair[1]

            # Each merge creates a new token.
            if merged_token in self.token_to_id:
                raise RuntimeError(
                    "Unexpected duplicate merge token."
                )

            new_id = len(self.token_to_id)

            self.token_to_id[merged_token] = new_id
            self.id_to_token[new_id] = merged_token

            self.merge_ranks[best_pair] = step

            vocab = self._merge_pair(best_pair, vocab)

            if verbose:
                print(
                    f"Merge {step + 1}: "
                    f"{best_pair[0]!r} + {best_pair[1]!r} "
                    f"-> {merged_token!r} "
                    f"(frequency={best_frequency})"
                )

        # Add special tokens after learned tokens.
        for token in self.special_tokens:

            token_id = len(self.token_to_id)

            self.special_to_id[token] = token_id
            self.id_to_special[token_id] = token

        self.is_trained = True

        self.validate()

        if verbose:
            print("\nTraining complete.")
            print("Vocabulary size:", self.vocabulary_size)
            print("Merge count:", len(self.merge_ranks))

    # ---------------------------------------------------------
    # 7. RANK-BASED MERGING
    # ---------------------------------------------------------

    def _apply_bpe(self, chunk: bytes) -> list[bytes]:
        """
        Apply learned merges according to their ranks.

        At each iteration, merge the available pair with
        the lowest learned rank.
        """

        symbols = [
            bytes([byte_value])
            for byte_value in chunk
        ]

        while len(symbols) > 1:

            best_rank = float("inf")
            best_pair = None

            for i in range(len(symbols) - 1):

                pair = (
                    symbols[i],
                    symbols[i + 1],
                )

                rank = self.merge_ranks.get(pair)

                if rank is not None and rank < best_rank:

                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            new_symbols = []

            i = 0

            while i < len(symbols):

                if (
                    i + 1 < len(symbols)
                    and symbols[i] == best_pair[0]
                    and symbols[i + 1] == best_pair[1]
                ):

                    new_symbols.append(
                        symbols[i] + symbols[i + 1]
                    )

                    i += 2

                else:

                    new_symbols.append(symbols[i])
                    i += 1

            symbols = new_symbols

        return symbols

    # ---------------------------------------------------------
    # 8. ENCODE
    # ---------------------------------------------------------

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        """
        Convert text into token IDs.

        Arbitrary Unicode input is supported through UTF-8 bytes.
        """

        self._check_trained()

        chunks = self._pre_tokenize(text)

        token_ids = []

        if add_special_tokens and "[BOS]" in self.special_to_id:
            token_ids.append(self.special_to_id["[BOS]"])

        for chunk in chunks:

            tokens = self._apply_bpe(chunk)

            for token in tokens:

                token_id = self.token_to_id.get(token)

                if token_id is None:
                    raise RuntimeError(
                        f"Missing vocabulary token: {token!r}"
                    )

                token_ids.append(token_id)

        if add_special_tokens and "[EOS]" in self.special_to_id:
            token_ids.append(self.special_to_id["[EOS]"])

        return token_ids

    # ---------------------------------------------------------
    # 9. DECODE
    # ---------------------------------------------------------

    def decode(
        self,
        token_ids: Iterable[int],
        skip_special_tokens: bool = True,
    ) -> str:

        self._check_trained()

        output = []
        byte_buffer = bytearray()

        def flush_bytes():
            if byte_buffer:
                output.append(
                    byte_buffer.decode("utf-8", errors="strict")
                )
                byte_buffer.clear()

        for token_id in token_ids:

            if not isinstance(token_id, int):
                raise TypeError("Token IDs must be integers.")

            if token_id in self.id_to_special:

                flush_bytes()

                if not skip_special_tokens:
                    output.append(self.id_to_special[token_id])

                continue

            token = self.id_to_token.get(token_id)

            if token is None:
                raise ValueError(f"Unknown token ID: {token_id}")

            byte_buffer.extend(token)

        flush_bytes()

        return "".join(output)

    # ---------------------------------------------------------
    # 10. TOKEN COUNT
    # ---------------------------------------------------------

    def count_tokens(self, text: str) -> int:

        return len(self.encode(text))

    # ---------------------------------------------------------
    # 11. VOCABULARY SIZE
    # ---------------------------------------------------------

    @property
    def vocabulary_size(self) -> int:

        return (
            len(self.token_to_id)
            + len(self.special_to_id)
        )

    # ---------------------------------------------------------
    # 12. VALIDATION
    # ---------------------------------------------------------

    def _check_trained(self) -> None:

        if not self.is_trained:
            raise RuntimeError(
                "Tokenizer has not been trained."
            )

    def validate(self) -> None:
        """
        Validate token ID consistency and merge configuration.
        """

        if len(self.token_to_id) != len(self.id_to_token):
            raise RuntimeError("Vocabulary mappings are inconsistent.")

        for token, token_id in self.token_to_id.items():

            if self.id_to_token.get(token_id) != token:
                raise RuntimeError("Invalid token ID mapping.")

        for pair, rank in self.merge_ranks.items():

            if len(pair) != 2:
                raise RuntimeError("Invalid merge pair.")

            merged = pair[0] + pair[1]

            if merged not in self.token_to_id:
                raise RuntimeError("Merge output missing from vocabulary.")

            if not isinstance(rank, int) or rank < 0:
                raise RuntimeError("Invalid merge rank.")

        for token, token_id in self.special_to_id.items():

            if self.id_to_special.get(token_id) != token:
                raise RuntimeError("Invalid special token mapping.")

    # ---------------------------------------------------------
    # 13. SAVE
    # ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save tokenizer configuration atomically.
        """

        self._check_trained()
        self.validate()

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        data = {
            "format_version": self.FORMAT_VERSION,
            "config": {
                "vocab_size": self.vocab_size,
                "min_frequency": self.min_frequency,
                "special_tokens": self.special_tokens,
            },
            "vocabulary": [
                {
                    "id": token_id,
                    "token_hex": token.hex(),
                }
                for token, token_id in self.token_to_id.items()
            ],
            "merges": [
                {
                    "left_hex": left.hex(),
                    "right_hex": right.hex(),
                    "rank": rank,
                }
                for (left, right), rank
                in self.merge_ranks.items()
            ],
            "special_token_ids": self.special_to_id,
        }

        temporary_path = None

        try:

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                delete=False,
            ) as file:

                temporary_path = Path(file.name)

                json.dump(
                    data,
                    file,
                    ensure_ascii=True,
                    indent=2,
                )

                file.flush()
                os.fsync(file.fileno())

            os.replace(temporary_path, path)

        finally:

            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()

    # ---------------------------------------------------------
    # 14. LOAD
    # ---------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "ByteLevelBPETokenizer":
        """
        Load and validate a previously saved tokenizer.
        """

        path = Path(path)

        with path.open(
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if data.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("Unsupported tokenizer format version.")

        config = data["config"]

        tokenizer = cls(
            vocab_size=config["vocab_size"],
            min_frequency=config["min_frequency"],
            special_tokens=config["special_tokens"],
        )

        # Reconstruct vocabulary.
        vocabulary = data["vocabulary"]

        token_to_id = {}
        id_to_token = {}

        for item in vocabulary:

            token_id = item["id"]
            token = bytes.fromhex(item["token_hex"])

            if token in token_to_id or token_id in id_to_token:
                raise ValueError("Duplicate vocabulary entry.")

            token_to_id[token] = token_id
            id_to_token[token_id] = token

        tokenizer.token_to_id = token_to_id
        tokenizer.id_to_token = id_to_token

        # Reconstruct merge ranks.
        tokenizer.merge_ranks = {}

        for item in data["merges"]:

            pair = (
                bytes.fromhex(item["left_hex"]),
                bytes.fromhex(item["right_hex"]),
            )

            rank = item["rank"]

            if pair in tokenizer.merge_ranks:
                raise ValueError("Duplicate merge rule.")

            tokenizer.merge_ranks[pair] = rank

        tokenizer.special_to_id = {
            token: int(token_id)
            for token, token_id
            in data["special_token_ids"].items()
        }

        tokenizer.id_to_special = {
            token_id: token
            for token, token_id
            in tokenizer.special_to_id.items()
        }

        tokenizer.is_trained = True

        tokenizer.validate()

        return tokenizer
