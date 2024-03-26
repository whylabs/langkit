from functools import partial

import pandas as pd

from langkit.core.metric import Metric, SingleMetric, SingleMetricResult, UdfInput
from langkit.metrics.embeddings_utils import compute_embedding_similarity
from langkit.transformer import embedding_adapter


def input_output_similarity_metric(input_column_name: str = "prompt", output_column_name: str = "response") -> Metric:
    def init():
        embedding_adapter()

    def udf(text: pd.DataFrame) -> SingleMetricResult:
        in_np = UdfInput(text).to_list(input_column_name)
        out_np = UdfInput(text).to_list(output_column_name)
        encoder = embedding_adapter()
        similarity = compute_embedding_similarity(encoder, in_np, out_np)

        if len(similarity.shape) == 1:
            return SingleMetricResult(similarity.tolist())  # type: ignore[reportUnknownVariableType]
        else:
            return SingleMetricResult(similarity.squeeze(dim=0).tolist())  # type: ignore[reportUnknownVariableType]

    return SingleMetric(
        name=f"{output_column_name}.similarity.{input_column_name}",
        input_names=[input_column_name, output_column_name],
        evaluate=udf,
        init=init,
    )


prompt_response_input_output_similarity_metric = partial(input_output_similarity_metric, "prompt", "response")
