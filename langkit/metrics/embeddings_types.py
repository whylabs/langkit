from functools import lru_cache
from typing import Protocol, Tuple

import torch
from sentence_transformers import SentenceTransformer


class TransformerEmbeddingAdapter:
    def __init__(self, transformer: SentenceTransformer):
        self._transformer = transformer

    @lru_cache(maxsize=6, typed=True)
    def encode(self, text: Tuple[str, ...]) -> "torch.Tensor":
        return torch.as_tensor(self._transformer.encode(sentences=list(text)))  # type: ignore[reportUnknownMemberType]


class EmbeddingEncoder(Protocol):
    def encode(self, text: Tuple[str, ...]) -> "torch.Tensor":
        ...


class CachingEmbeddingEncoder(EmbeddingEncoder):
    def __init__(self, transformer: EmbeddingEncoder):
        self._transformer = transformer

    @lru_cache(maxsize=6, typed=True)
    def encode(self, text: Tuple[str, ...]) -> "torch.Tensor":  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._transformer.encode(text)  # type: ignore[no-any-return]
