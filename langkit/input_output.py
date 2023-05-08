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


def register_udf(name: str, func: Callable, col_type: DataType):
    _col_type_submetrics[col_type].append((name, func))


# requires %pip install sentence-transformers -q
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def get_subject_similarity(text: str, comparison_embedding: Tensor) -> float:
    embedding = model.encode(text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding, comparison_embedding)
    return similarity.item()


unpleasant_subjects = ["legal inquiry", "politics"]

for unpleasant_subject in unpleasant_subjects:
    unpleasant_subject_embedding = model.encode(unpleasant_subject, convert_to_tensor=True)
    register_udf(f"topic_{unpleasant_subject.replace(' ', '_')}",
                 lambda text, subject=unpleasant_subject_embedding: get_subject_similarity(text, subject), String)


@register_metric_udf(col_name="combined")
def similarity_MiniLM_L6_v2(combined: Tuple[str, str]) -> float:
    x, y = combined
    embedding_1 = model.encode(x, convert_to_tensor=True)
    embedding_2 = model.encode(y, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
    return similarity.item()
