from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from langkit.core.context import Context, ContextDependency
from langkit.metrics.embeddings_types import EmbeddingEncoder, TransformerEmbeddingAdapter
from langkit.onnx_encoder import OnnxSentenceTransformer, TransformerModel


def _sentence_transformer(
    name_revision: Tuple[str, str] = ("all-MiniLM-L6-v2", "44eb4044493a3c34bc6d7faae1a71ec76665ebc6"),
) -> SentenceTransformer:
    """
    Returns a SentenceTransformer model instance.

    The intent of this function is to cache the SentenceTransformer instance to avoid
    multple instances being created all over langkit, and have a single place that
    can be used to change the transformer name for the metrics that default to the same one.
    """
    transformer_name, revision = name_revision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(transformer_name, revision=revision, device=device)


@lru_cache
def embedding_adapter(onnx: bool = True) -> EmbeddingEncoder:
    if onnx:
        return OnnxSentenceTransformer(TransformerModel.AllMiniLM)
    else:
        return TransformerEmbeddingAdapter(_sentence_transformer())


@dataclass(frozen=True)
class EmbeddingContextDependency(ContextDependency[torch.Tensor]):
    onnx: bool
    input_column: str

    def name(self) -> str:
        return f"{self.input_column}.embedding?onnx={self.onnx}"

    def cache_assets(self) -> None:
        # TODO do only the downloading
        embedding_adapter(onnx=self.onnx)

    def init(self) -> None:
        embedding_adapter(onnx=self.onnx)

    def populate_request(self, context: Context, data: pd.DataFrame):
        if self.input_column not in data.columns:
            return

        if self.name() in context.request_data:
            return

        encoder = embedding_adapter(onnx=self.onnx)
        embedding = encoder.encode(tuple(data[self.input_column]))  # pyright: ignore[reportUnknownArgumentType]
        context.request_data[self.name()] = embedding

    def get_request_data(self, context: Context) -> torch.Tensor:
        return context.request_data[self.name()]
