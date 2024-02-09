from typing import List, Protocol

import torch
from sentence_transformers import SentenceTransformer


class TransformerEmbeddingAdapter:
    def __init__(self, transformer: SentenceTransformer):
        self._transformer = transformer

    def encode(self, text: List[str]) -> "torch.Tensor":
        assert isinstance(text, list)
        return torch.as_tensor(self._transformer.encode(sentences=text))  # type: ignore[reportUnknownMemberType]


class EmbeddingEncoder(Protocol):
    def encode(self, text: List[str]) -> "torch.Tensor":
        ...
