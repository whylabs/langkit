from typing import Optional, Tuple
from sentence_transformers import SentenceTransformer, util
from torch import Tensor
from whylogs.experimental.core.metrics.udf_metric import (
    register_metric_udf,
)

_transformer_model = None


def init(transformer_name: Optional[str]):
    global _transformer_model
    if transformer_name is None:
        transformer_name = 'sentence-transformers/all-MiniLM-L6-v2'
    _transformer_model = SentenceTransformer(transformer_name)


init()


def get_subject_similarity(text: str, comparison_embedding: Tensor) -> float:
    embedding = _transformer_model.encode(text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding, comparison_embedding)
    return similarity.item()


@register_metric_udf(col_name="combined")
def similarity_MiniLM_L6_v2(combined: Tuple[str, str]) -> float:
    x, y = combined
    embedding_1 = _transformer_model.encode(x, convert_to_tensor=True)
    embedding_2 = _transformer_model.encode(y, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
    return similarity.item()
