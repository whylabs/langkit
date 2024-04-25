from typing import Protocol, Tuple

import torch
from sentence_transformers import SentenceTransformer


class EmbeddingEncoder(Protocol):
    def encode(self, text: Tuple[str, ...]) -> "torch.Tensor":
        ...


class TransformerEmbeddingAdapter(EmbeddingEncoder):
    def __init__(self, transformer: SentenceTransformer):
        self._transformer = transformer

    def encode(self, text: Tuple[str, ...]) -> "torch.Tensor":  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._transformer.encode(sentences=list(text), convert_to_tensor=True, show_progress_bar=False)  # type: ignore[reportUnknownMemberType]
