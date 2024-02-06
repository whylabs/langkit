from functools import partial
from typing import List, Optional

import pandas as pd
import torch
import torch.nn.functional as F

from langkit.core.metric import Metric, SingleMetric, SingleMetricResult, UdfInput
from langkit.metrics.input_output_similarity_types import EmbeddingEncoder, TransformerEmbeddingAdapter
from langkit.transformer import sentence_transformer


def __compute_embedding_similarity(encoder: EmbeddingEncoder, _in: List[str], _out: List[str]) -> torch.Tensor:
    in_encoded = torch.as_tensor(encoder.encode(_in))
    out_encoded = torch.as_tensor(encoder.encode(_out))
    sim = F.cosine_similarity(in_encoded, out_encoded, dim=1)
    return sim


def input_output_similarity_metric(
    input_column_name: str = "prompt", output_column_name: str = "response", embedding_encoder: Optional[EmbeddingEncoder] = None
) -> Metric:
    transformer_name = "sentence-transformers/all-MiniLM-L6-v2"
    encoder = embedding_encoder or TransformerEmbeddingAdapter(sentence_transformer.value(transformer_name))

    def init():
        sentence_transformer.value(transformer_name)

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
