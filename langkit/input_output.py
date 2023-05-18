from typing import Optional, Tuple
from sentence_transformers import SentenceTransformer, util
from torch import Tensor
from whylogs.experimental.core.udf_schema import (
    generate_udf_dataset_schema,
    register_dataset_udf,
)
from . import LangKitConfig
from langkit.transformer import load_model

lang_config = LangKitConfig()


_transformer_model = None
_transformer_name = None


def init(transformer_name: Optional[str]=None):
    global _transformer_model, _transformer_name
    if transformer_name is None:
        transformer_name = lang_config.transformer_name
    _transformer_model = load_model(transformer_name)
    _transformer_name = transformer_name

init()


@register_dataset_udf(["prompt", "response"], _transformer_name)
def similarity_MiniLM_L6_v2(text):
    x = text["prompt"]
    y = text["response"]
    # below assumes text is Dict[str, str], no pandas support
    embedding_1 = _transformer_model.encode(x, convert_to_tensor=True)
    embedding_2 = _transformer_model.encode(y, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
    return similarity.item()
