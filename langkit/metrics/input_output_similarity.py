from functools import partial
from typing import List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from langkit.core.metric import Metric, SingleMetric, SingleMetricResult, UdfInput
from langkit.metrics.input_output_similarity_types import EmbeddingEncoder, TransformerEmbeddingAdapter
from langkit.metrics.util import LazyInit


def __compute_embedding_similarity(encoder: EmbeddingEncoder, _in: List[str], _out: List[str]) -> torch.Tensor:
    in_encoded = torch.as_tensor(encoder.encode(_in))
    out_encoded = torch.as_tensor(encoder.encode(_out))
    print(f"computing similarities between {in_encoded.shape} and {out_encoded.shape}")
    sim = F.cosine_similarity(in_encoded, out_encoded, dim=1)

    print(f"computed similarity: {sim}")
    return sim


__transformer = LazyInit(
    lambda: SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
)


def input_output_similarity_metric(
    input_column_name: str = "prompt", output_column_name: str = "response", embedding_encoder: Optional[EmbeddingEncoder] = None
) -> Metric:
    encoder = embedding_encoder or TransformerEmbeddingAdapter(__transformer.value)

    def init():
        __transformer.value

    def udf(text: pd.DataFrame) -> SingleMetricResult:
        in_np = UdfInput(text).to_list(input_column_name)
        out_np = UdfInput(text).to_list(output_column_name)
        similarity = __compute_embedding_similarity(encoder, in_np, out_np)

        if len(similarity.shape) == 1:
            return SingleMetricResult(similarity.tolist())  # type: ignore[reportUnknownVariableType]
        else:
            return SingleMetricResult(similarity.squeeze(dim=0).tolist())  # type: ignore[reportUnknownVariableType]

    return SingleMetric(
        name=f"{output_column_name}.relevance_to_{input_column_name}",
        input_name=input_column_name,
        evaluate=udf,
        init=init,
    )


prompt_response_input_output_similarity_module = partial(input_output_similarity_metric, "prompt", "response", None)
