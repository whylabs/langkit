from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Literal, Union

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from langkit.core.context import Context, ContextDependency
from langkit.core.workflow import InputContext
from langkit.metrics.embeddings_types import EmbeddingEncoder, TransformerEmbeddingAdapter
from langkit.onnx_encoder import OnnxSentenceTransformer, TransformerModel


class EmbeddingChoice(ABC):
    @abstractmethod
    def get_encoder(self) -> EmbeddingEncoder:
        raise NotImplementedError()


class SentenceTransformerChoice(EmbeddingChoice):
    def __init__(self, name: str, revision: str):
        self.name = name
        self.revision = revision

    def get_encoder(self) -> EmbeddingEncoder:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return TransformerEmbeddingAdapter(SentenceTransformer(self.name, revision=self.revision, device=device))


class DefaultChoice(SentenceTransformerChoice):
    def __init__(self):
        super().__init__("all-MiniLM-L6-v2", "44eb4044493a3c34bc6d7faae1a71ec76665ebc6")


class OnnxChoice(EmbeddingChoice):
    def get_encoder(self) -> EmbeddingEncoder:
        return OnnxSentenceTransformer(TransformerModel.AllMiniLM)


@dataclass(frozen=True)
class SentenceTransformerTarget:
    name: str
    revision: str


EmbeddingChoiceArg = Union[Literal["default"], Literal["onnx"], SentenceTransformerTarget]


@lru_cache
def embedding_adapter(choice: EmbeddingChoiceArg = "default") -> EmbeddingEncoder:
    if choice == "default":
        return DefaultChoice().get_encoder()
    elif choice == "onnx":
        return OnnxChoice().get_encoder()
    else:
        return SentenceTransformerChoice(choice.name, choice.revision).get_encoder()


@dataclass(frozen=True)
class EmbeddingContextDependency(ContextDependency[torch.Tensor]):
    embedding_choice: EmbeddingChoiceArg
    input_column: str

    def name(self) -> str:
        if self.embedding_choice == "default":
            choice_str = "default"
        elif self.embedding_choice == "onnx":
            choice_str = "onnx"
        else:
            choice_str = f"{self.embedding_choice.name}-{self.embedding_choice.revision}"

        return f"{self.input_column}.embedding?type={choice_str}"

    def _get_encoder(self) -> EmbeddingEncoder:
        return embedding_adapter(choice=self.embedding_choice)

    def cache_assets(self) -> None:
        self._get_encoder()

    def init(self) -> None:
        self._get_encoder()

    def populate_request(self, context: Context, data: pd.DataFrame):
        if self.input_column not in data.columns:
            return

        if self.name() in context.request_data:
            return

        encoder = self._get_encoder()
        embedding = encoder.encode(tuple(data[self.input_column]))  # pyright: ignore[reportUnknownArgumentType]
        context.request_data[self.name()] = embedding

    def get_request_data(self, context: Context) -> torch.Tensor:
        return context.request_data[self.name()]


@dataclass(frozen=True)
class RAGContextDependency(ContextDependency[torch.Tensor]):
    embedding_choice: EmbeddingChoiceArg
    strategy: Literal["combine"] = "combine"
    """
    The strategy for converting the context into embeddings.

    - combine: Combine all the entries in the context into a single string and encode it.
    """
    context_column_name: str = "context"

    def name(self) -> str:
        if self.embedding_choice == "default":
            choice_str = "default"
        elif self.embedding_choice == "onnx":
            choice_str = "onnx"
        else:
            choice_str = f"{self.embedding_choice.name}-{self.embedding_choice.revision}"

        return f"{self.context_column_name}.context?type={choice_str}&strategy={self.strategy}"

    def _get_encoder(self) -> EmbeddingEncoder:
        return embedding_adapter(choice=self.embedding_choice)

    def cache_assets(self) -> None:
        self._get_encoder()

    def init(self) -> None:
        self._get_encoder()

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

        encoder = self._get_encoder()
        embedding = encoder.encode(tuple(combined))
        context.request_data[self.name()] = embedding

    def _get_rag_context(self, df: pd.DataFrame) -> List[InputContext]:
        context_column: List[InputContext] = df[self.context_column_name].tolist()
        return context_column

    def get_request_data(self, context: Context) -> torch.Tensor:
        return context.request_data[self.name()]
