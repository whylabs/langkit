from typing import Tuple
from sentence_transformers import SentenceTransformer, util
from whylogs.experimental.core.metrics.udf_metric import (
    register_metric_udf,
)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


@register_metric_udf(col_name="combined")
def similarity_MiniLM_L6_v2(combined: Tuple[str, str]) -> float:
    x, y = combined
    embedding_1 = model.encode(x, convert_to_tensor=True)
    embedding_2 = model.encode(y, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
    return similarity.item()
