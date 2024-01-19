from typing import Any, List, Protocol, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingEncoder(Protocol):
    def encode(self, text: List[str]) -> Union[torch.Tensor, np.ndarray[Any, Any]]:
        ...


class TransformerEmbeddingAdapter(EmbeddingEncoder):
    def __init__(self, transformer: SentenceTransformer):
        self._transformer = transformer

    def encode(self, text: List[str]) -> torch.Tensor:
        assert isinstance(text, list)
        return torch.as_tensor(self._transformer.encode(sentences=text))  # type: ignore[reportUnknownMemberType]
