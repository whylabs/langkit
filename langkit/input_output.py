from typing import Tuple
from sentence_transformers import SentenceTransformer, util
from torch import Tensor
from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import (
    register_metric_udf,
)
from typing import Callable
from whylogs.core.datatypes import DataType
from whylogs.experimental.core.metrics.udf_metric import (
    _col_type_submetrics
)

_example_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def register_udf(name: str, func: Callable, col_type: DataType):
    _col_type_submetrics[col_type].append((name, func))


def get_subject_similarity(text: str, comparison_embedding: Tensor) -> float:
    embedding = _example_model.encode(text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding, comparison_embedding)
    return similarity.item()


@register_metric_udf(col_name="combined")
def similarity_MiniLM_L6_v2(combined: Tuple[str, str]) -> float:
    x, y = combined
    embedding_1 = _example_model.encode(x, convert_to_tensor=True)
    embedding_2 = _example_model.encode(y, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
    return similarity.item()
