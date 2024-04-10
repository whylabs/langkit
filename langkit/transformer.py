from dataclasses import dataclass
from functools import lru_cache
from typing import List, Literal, Tuple

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from langkit.core.context import Context, ContextDependency
from langkit.core.workflow import InputContext
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


@dataclass(frozen=True)
class RAGContextDependency(ContextDependency[torch.Tensor]):
    onnx: bool
    strategy: Literal["combine"] = "combine"
    """
    The strategy for converting the context into embeddings.

    - combine: Combine all the entries in the context into a single string and encode it.
    """
    context_column_name: str = "context"

    def name(self) -> str:
        return f"{self.context_column_name}.context?onnx={self.onnx}"

    def cache_assets(self) -> None:
        # TODO do only the downloading
        embedding_adapter(onnx=self.onnx)

    def init(self) -> None:
        embedding_adapter(onnx=self.onnx)

    def populate_request(self, context: Context, data: pd.DataFrame):
        if self.context_column_name not in data.columns:
            return

        if self.name() in context.request_data:
            return

        rag_context = self._get_rag_context(data)

        if self.strategy == "combine":
            combined: List[str] = []
            for row in rag_context:
                print(row)
                row_string = "\n".join([it["content"] for it in row["entries"]])
                combined.append(row_string)
        else:
            raise ValueError(f"Unknown context embedding strategy {self.strategy}")

        encoder = embedding_adapter(onnx=self.onnx)
        embedding = encoder.encode(tuple(combined))
        context.request_data[self.name()] = embedding

    def _get_rag_context(self, df: pd.DataFrame) -> List[InputContext]:
        context_column: List[InputContext] = df[self.context_column_name].tolist()
        return context_column

    def get_request_data(self, context: Context) -> torch.Tensor:
        return context.request_data[self.name()]
