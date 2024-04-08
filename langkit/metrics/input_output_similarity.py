from functools import partial

import pandas as pd

from langkit.core.context import Context
from langkit.core.metric import Metric, SingleMetric, SingleMetricResult
from langkit.metrics.embeddings_utils import compute_embedding_similarity_encoded
from langkit.transformer import EmbeddingContextDependency


def input_output_similarity_metric(input_column_name: str = "prompt", output_column_name: str = "response", onnx: bool = True) -> Metric:
    prompt_embedding_dep = EmbeddingContextDependency(onnx=onnx, input_column=input_column_name)
    response_embedding_dep = EmbeddingContextDependency(onnx=onnx, input_column=output_column_name)

    def udf(text: pd.DataFrame, context: Context) -> SingleMetricResult:
        # in_np = UdfInput(text).to_list(input_column_name)
        # out_np = UdfInput(text).to_list(output_column_name)
        # encoder = embedding_adapter(onnx)
        prompt_embedding = prompt_embedding_dep.get_request_data(context)
        response_embedding = response_embedding_dep.get_request_data(context)
        similarity = compute_embedding_similarity_encoded(prompt_embedding, response_embedding)

        if len(similarity.shape) == 1:
            return SingleMetricResult(similarity.tolist())  # type: ignore[reportUnknownVariableType]
        else:
            return SingleMetricResult(similarity.squeeze(dim=0).tolist())  # type: ignore[reportUnknownVariableType]

    return SingleMetric(
        name=f"{output_column_name}.similarity.{input_column_name}",
        input_names=[input_column_name, output_column_name],
        evaluate=udf,
        context_dependencies=[prompt_embedding_dep, response_embedding_dep],
    )


prompt_response_input_output_similarity_metric = partial(input_output_similarity_metric, "prompt", "response")
