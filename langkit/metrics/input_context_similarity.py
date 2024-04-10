import pandas as pd

from langkit.core.context import Context
from langkit.core.metric import Metric, SingleMetric, SingleMetricResult
from langkit.metrics.embeddings_utils import compute_embedding_similarity_encoded
from langkit.transformer import EmbeddingContextDependency, RAGContextDependency


def input_context_similarity(input_column_name: str = "prompt", context_column_name: str = "context", onnx: bool = True) -> Metric:
    prompt_embedding_dep = EmbeddingContextDependency(onnx=onnx, input_column=input_column_name)
    context_embedding_dep = RAGContextDependency(onnx=onnx, context_column_name=context_column_name)

    def udf(text: pd.DataFrame, context: Context) -> SingleMetricResult:
        prompt_embedding = prompt_embedding_dep.get_request_data(context)
        context_embedding = context_embedding_dep.get_request_data(context)
        similarity = compute_embedding_similarity_encoded(prompt_embedding, context_embedding)

        if len(similarity.shape) == 1:
            return SingleMetricResult(similarity.tolist())  # type: ignore[reportUnknownVariableType]
        else:
            return SingleMetricResult(similarity.squeeze(dim=0).tolist())  # type: ignore[reportUnknownVariableType]

    return SingleMetric(
        name=f"{input_column_name}.similarity.{context_column_name}",
        input_names=[input_column_name, context_column_name],
        evaluate=udf,
        context_dependencies=[prompt_embedding_dep, context_embedding_dep],
    )
