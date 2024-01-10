from functools import partial
from typing import Any, List, Optional, Protocol, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from langkit.module.module import EvaluationResult, MetricConfig, UdfInput


class EmbeddingEncoder(Protocol):
    def encode(self, text: List[str]) -> Union[torch.Tensor, np.ndarray[Any, Any]]:
        ...


class TransformerEmbeddingAdapter:
    def __init__(self, transformer: SentenceTransformer):
        self._transformer = transformer

    def encode(self, text: List[str]) -> torch.Tensor:
        assert isinstance(text, list)
        return torch.as_tensor(self._transformer.encode(sentences=text))  # type: ignore[reportUnknownMemberType]


def __compute_embedding_similarity(encoder: EmbeddingEncoder, _in: List[str], _out: List[str]) -> torch.Tensor:
    in_encoded = torch.as_tensor(encoder.encode(_in))
    out_encoded = torch.as_tensor(encoder.encode(_out))
    print(f"computing similarities between {in_encoded.shape} and {out_encoded.shape}")
    sim = F.cosine_similarity(in_encoded, out_encoded, dim=1)

    print(f"computed similarity: {sim}")
    return sim


def __input_output_similarity_module(
    input_column_name: str = "prompt", output_column_name: str = "response", embedding_encoder: Optional[EmbeddingEncoder] = None
) -> MetricConfig:
    if embedding_encoder is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = TransformerEmbeddingAdapter(SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device))
    else:
        encoder = embedding_encoder

    def udf(text: pd.DataFrame) -> EvaluationResult:
        in_np = UdfInput(text).to_list(input_column_name)
        out_np = UdfInput(text).to_list(output_column_name)
        similarity = __compute_embedding_similarity(encoder, in_np, out_np)

        if len(similarity.shape) == 1:
            return EvaluationResult(similarity.tolist())  # type: ignore[reportUnknownVariableType]
        else:
            return EvaluationResult(similarity.squeeze(dim=0).tolist())  # type: ignore[reportUnknownVariableType]

    return MetricConfig(
        name=f"{output_column_name}.relevance_to_{input_column_name}",
        input_name=input_column_name,
        evaluate=udf,
    )


input_output_similarity_module = partial(__input_output_similarity_module, "prompt", "response", None)
