from functools import lru_cache
from typing import Protocol, Tuple

import torch
from sentence_transformers import SentenceTransformer


class EmbeddingEncoder(Protocol):
    def encode(self, text: Tuple[str, ...]) -> "torch.Tensor":
        ...


class TransformerEmbeddingAdapter(EmbeddingEncoder):
    def __init__(self, transformer: SentenceTransformer):
        self._transformer = transformer

    @lru_cache(maxsize=6, typed=True)
    def encode(self, text: Tuple[str, ...]) -> "torch.Tensor":  # pyright: ignore[reportIncompatibleMethodOverride]
        return torch.as_tensor(self._transformer.encode(sentences=list(text), show_progress_bar=False))  # type: ignore[reportUnknownMemberType]


class CachingEmbeddingEncoder(EmbeddingEncoder):
    def __init__(self, transformer: EmbeddingEncoder):
        self._transformer = transformer

    @lru_cache(maxsize=6, typed=True)
    def encode(self, text: Tuple[str, ...]) -> "torch.Tensor":  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._transformer.encode(text)  # type: ignore[no-any-return]
