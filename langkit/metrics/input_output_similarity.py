from functools import partial
from typing import Optional

import pandas as pd

from langkit.core.metric import Metric, SingleMetric, SingleMetricResult, UdfInput
from langkit.metrics.embeddings_types import EmbeddingEncoder, TransformerEmbeddingAdapter
from langkit.metrics.embeddings_utils import compute_embedding_similarity
from langkit.transformer import sentence_transformer


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
        similarity = compute_embedding_similarity(encoder, in_np, out_np)

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
