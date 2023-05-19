from typing import Optional

from sentence_transformers import util
from whylogs.experimental.core.udf_schema import register_dataset_udf

from langkit.transformer import load_model

from . import LangKitConfig

lang_config = LangKitConfig()
_transformer_model = None
_transformer_name = None


def init(transformer_name: Optional[str] = None):
    global _transformer_model, _transformer_name
    if transformer_name is None:
        transformer_name = lang_config.transformer_name
    _transformer_model = load_model(transformer_name)
    _transformer_name = transformer_name


init()
_udf_name = _transformer_name or lang_config.transformer_name


@register_dataset_udf(["prompt", "response"], f"similarity_{_udf_name.split('/')[-1]}")
def similarity_MiniLM_L6_v2(text):
    x = text["prompt"]
    y = text["response"]
    # below assumes text is Dict[str, str], no pandas support
    embedding_1 = _transformer_model.encode(x, convert_to_tensor=True)
    embedding_2 = _transformer_model.encode(y, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
    return similarity.item()
