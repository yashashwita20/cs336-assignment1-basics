import heapq
import os
import pickle
from collections import Counter
from collections.abc import Iterable, Iterator
from multiprocessing.pool import Pool

import regex as re

# GPT-2 Pattern fetched from github.com/openai/tiktoken/pull/234/files:
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class RevBytes:
    def __init__(self, b: bytes):
        self.b = b

    def __lt__(self, other):
        return self.b > other.b

    def __eq__(self, other):
        return self.b == other.b


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ) -> None:
        """
        A BPE tokenizer implementation.
        Args:
            vocab: dict[int, bytes] - Mapping from token ID to byte string
            merges: list[tuple[bytes, bytes]] - List of merges (pairs of byte strings)
            special_tokens: list[str] | None - Optional list of special tokens
        """
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        self._encode_cache = {}
        self.pretokenize_pattern = re.compile(PAT)

        if self.special_tokens:
            # Add new special token to vocabulary
            next_id = max(self.vocab.keys()) + 1
            for token in special_tokens:
                token_bytes = token.encode("UTF-8")
                if token_bytes not in self.vocab_inv:
                    self.vocab[next_id] = token_bytes
                    self.vocab_inv[token_bytes] = next_id
                    next_id += 1

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        Class method to construct a Tokenizer from vocab and merges files.
        Args:
            vocab_filepath: Path to vocab file (token ID to byte string)
            merges_filepath: Path to merges file (list of byte string pairs)
            special_tokens: Optional list of special tokens
        Returns:
            An instance of Tokenizer
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        encoded = []

        text_parts = _split_on_special_tokens(text, self.special_tokens)

        for part in text_parts:
            if part in self.special_tokens:
                # Encode special token directly
                token_id = self.vocab_inv[part.encode("utf-8")]
                encoded.append(token_id)
            else:
                # Pre-tokenization
                pre_tokens = self.pretokenize_pattern.findall(part)

                for pre_token in pre_tokens:
                    if pre_token in self._encode_cache:
                        encoded.extend(self._encode_cache[pre_token])
                    else:
                        byte_list = [bytes([b]) for b in pre_token.encode("utf-8")]

                        while len(byte_list) > 1:
                            byte_pairs = get_byte_pairs(byte_list)

                            # find the pair with the lowest merge index
                            # if no pairs are in merges, then all pairs will have rank inf
                            # min will be the first pair in that case, so we check if it's in merge_ranks and break out of the loop
                            best_pair = min(byte_pairs, key=lambda pair: self.merge_ranks.get(pair, float("inf")))

                            if best_pair not in self.merge_ranks:
                                break  # no more merges to apply

                            # else merge the best pair available
                            new_byte_list = []
                            i = 0

                            while i < len(byte_list):
                                if i < len(byte_list) - 1 and (byte_list[i], byte_list[i + 1]) == best_pair:
                                    new_byte_list.append(best_pair[0] + best_pair[1])  # merge the pair
                                    i += 2  # skip the next byte since it's merged
                                else:
                                    new_byte_list.append(byte_list[i])
                                    i += 1
                            byte_list = new_byte_list
                        token_ids = [self.vocab_inv[b] for b in byte_list]
                        encoded.extend(token_ids)
                        self._encode_cache[pre_token] = token_ids

        return encoded

    def encode_iterable(self, iterable: "Iterable[str]") -> "Iterator[int]":
        """Given an iterable of strings (e.g., a Python file handle),
        return a generator that lazily yields token IDs. This is required
        for memory-efficient tokenization of large files that we cannot
        directly load into memory."""
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text"""
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text


def get_byte_pairs(ids: list, counts: dict[tuple[int, int], int] = None) -> dict[tuple[int, int], int]:
    """Given a list of token IDs and a dict to store counts, return a dict of byte pair frequencies."""
    counts = {} if counts is None else counts

    for pair in zip(ids[:-1], ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _split_on_special_tokens(text: str, special_tokens: list[str] = None) -> list[str]:
    if not special_tokens:
        return [text]

    # Sort special tokens by length (descending) to ensure greedy matching
    special_tokens = sorted(special_tokens, key=len, reverse=True)

    # escape special chars in tokens like <|endoftext|> for regex and join with
    escaped_tokens = "|".join(re.escape(t) for t in special_tokens)

    # regex.split with capture group keeps the delimiters
    split_pattern = re.compile(f"({escaped_tokens})")

    parts = split_pattern.split(text)

    return [p for p in parts if p]  # remove empty strings


def find_chunk_boundaries(f, special_tokens, num_workers):
    special_tokens_bytes = [t.encode("utf-8") for t in special_tokens]

    f.seek(0, os.SEEK_END)
    file_size = f.tell()
    f.seek(0)

    # No special tokens → nowhere safe to split, use one chunk
    if not special_tokens_bytes:
        return [0, file_size]

    chunk_size = file_size // num_workers
    boundaries = [i * chunk_size for i in range(num_workers + 1)]
    boundaries[-1] = file_size

    mini_chunk_size = 4096
    max_token_len = max(len(t) for t in special_tokens_bytes)

    for bi in range(1, len(boundaries) - 1):
        pos = boundaries[bi]
        f.seek(pos)
        while True:
            mini_chunk = f.read(mini_chunk_size)
            if mini_chunk == b"":
                boundaries[bi] = file_size
                break

            # earliest occurrence of any special token in this read
            hits = [i for t in special_tokens_bytes if (i := mini_chunk.find(t)) != -1]
            if hits:
                boundaries[bi] = pos + min(hits)
                break

            # overlap so a token cut across two reads isn't missed
            pos += max(len(mini_chunk) - max_token_len + 1, 1)
            f.seek(pos)

    return sorted(set(boundaries))


def _pretokenize_chunk(chunk: str, special_tokens: list[str]) -> Counter:
    pretoken_frequencies = Counter()

    # split the chunk into pieces, keeping special tokens as their own pieces
    for part in _split_on_special_tokens(chunk, special_tokens):
        if part in special_tokens:
            continue  # don't count special tokens, and never merge across them

        for pre_token in re.finditer(PAT, part):
            byte_tuple = tuple(bytes([b]) for b in pre_token.group().encode("utf-8"))
            pretoken_frequencies[byte_tuple] += 1

    return pretoken_frequencies


def pretokenize(
    input_path: str | os.PathLike, special_tokens: list[str] | None = None, num_workers: int = None
) -> list:
    """Pre-tokenize the input file by splitting on special tokens and pretokenization regex."""

    if num_workers is None:
        num_workers = os.cpu_count()

    special_tokens = special_tokens or []

    chunk_pretoken_frequencies = (
        Counter()
    )  # Counter of pre-tokens across the whole file, to be updated in parallel by each worker

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, special_tokens, num_workers)

        async_results = []
        with Pool(num_workers) as pool:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                async_results.append(pool.apply_async(_pretokenize_chunk, (chunk, special_tokens)))

            for async_result in async_results:
                chunk_pretoken_frequencies.update(async_result.get())

            pool.close()  # no more tasks will be submitted to the pool
            pool.join()  # wait for all processes to finish

    return chunk_pretoken_frequencies


def build_pair_index(
    pretoken_frequencies: Counter,
) -> tuple[Counter, dict[tuple[bytes, bytes], set[bytes]], list[tuple[int, tuple[bytes, bytes]]]]:
    pair_freq = Counter()
    pair_to_words = {}
    for word, freq in pretoken_frequencies.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freq[pair] += freq
            pair_to_words.setdefault(pair, set()).add(word)

    heap = [
        (-freq, RevBytes(p[0]), RevBytes(p[1])) for p, freq in pair_freq.items()
    ]  # max heap based on frequency, with pair as tie-breaker and negative for min-heap behavior
    heapq.heapify(heap)

    return pair_freq, pair_to_words, heap


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str] | None = None, verbose=False
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given a text file, train a (byte-level) BPE tokenizer and return the vocab and merges."""

    if vocab_size < 256:
        raise ValueError("Vocab size must be at least 256.")

    # Initialize Vocab
    vocab = {}
    next_id = 0

    if special_tokens:
        for token in special_tokens:
            token_bytes = token.encode("utf-8")
            vocab[next_id] = token_bytes
            next_id += 1

    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1

    num_merges = vocab_size - next_id  # 256 byte tokens + special tokens in vocab

    if verbose:
        print(f"Initialized vocab with {len(vocab)} tokens (including special tokens).")
        print(f"Training BPE with target vocab size {vocab_size} ({num_merges} merges to perform).")
        print(f"Reading input file: {input_path}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    pretoken_frequencies = pretokenize(input_path, special_tokens, num_workers=os.cpu_count())
    pair_freq, pair_to_words, heap = build_pair_index(pretoken_frequencies)

    merges = []
    for _ in range(num_merges):
        while heap:
            neg_freq, p1, p2 = heapq.heappop(heap)
            pair = (p1.b, p2.b)
            if -neg_freq == pair_freq.get(pair, 0):
                break
        else:
            if verbose:
                print("No more pairs to merge.")
            break

        merges.append(pair)
        if verbose:
            print(f"Merge {len(merges)}: {pair} (freq: {-neg_freq})")
        new_token = pair[0] + pair[1]
        vocab[next_id] = new_token
        next_id += 1
        affected_words = pair_to_words.get(pair, set())

        for word in list(affected_words):
            freq = pretoken_frequencies[word]
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_tuple = tuple(new_word)

            del pretoken_frequencies[word]  # remove old word frequency
            pretoken_frequencies[new_word_tuple] += freq  # update frequency of new word

            # update byte_pair_to_words: remove old, add new
            for i in range(len(word) - 1):
                pair_to_words.setdefault((word[i], word[i + 1]), set()).discard(word)
            for i in range(len(new_word_tuple) - 1):
                pair_to_words.setdefault((new_word_tuple[i], new_word_tuple[i + 1]), set()).add(new_word_tuple)

            # update byte_pair_freq: diff old pairs vs new pairs for this word
            old_pairs = Counter(zip(word, word[1:]))
            new_pairs = Counter(zip(new_word_tuple, new_word_tuple[1:]))
            for p in set(old_pairs) | set(new_pairs):
                delta = (new_pairs.get(p, 0) - old_pairs.get(p, 0)) * freq
                if delta:
                    pair_freq[p] += delta
                    if pair_freq[p] <= 0:
                        del pair_freq[p]
                    else:
                        heapq.heappush(heap, (-pair_freq[p], RevBytes(p[0]), RevBytes(p[1])))

    return vocab, merges
