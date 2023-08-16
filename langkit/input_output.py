from logging import getLogger
from typing import Callable, Optional

from sentence_transformers import util
from whylogs.experimental.core.udf_schema import register_dataset_udf

from langkit.transformer import load_model

from . import LangKitConfig

try:
    import tensorflow as tf
except ImportError:
    tf = None

lang_config = LangKitConfig()
_transformer_model = None
_transformer_name = None

diagnostic_logger = getLogger(__name__)

class CustomEncoder:
    def __init__(self, encoder: Callable):
        self.encode = encoder

def init(transformer_name: Optional[str] = None, encoder: Optional[Callable] = None):
    global _transformer_model, _transformer_name
    if transformer_name and encoder:
        raise ValueError(
            "Only one of transformer_name or encoder can be specified, not both."
        )
    if encoder:
        _transformer_model = CustomEncoder(encoder)
        return

    if transformer_name is None and encoder is None:
        transformer_name = lang_config.transformer_name
    _transformer_model = load_model(transformer_name)
    _transformer_name = transformer_name


init()


@register_dataset_udf(["prompt", "response"], "response.relevance_to_prompt")
def prompt_response_similarity(text):
    if _transformer_model is None:
        raise ValueError(
            "response.relevance_to_prompt must have a transformer model initialized before use."
        )

    series_result = []
    for x, y in zip(text["prompt"], text["response"]):
        try:
            if not isinstance(_transformer_model,CustomEncoder):
                embedding_1 = _transformer_model.encode([x], convert_to_tensor=True)
                embedding_2 = _transformer_model.encode([y], convert_to_tensor=True)
            else:
                embedding_1 = _transformer_model.encode([x])
                embedding_2 = _transformer_model.encode([y])
            if tf and isinstance(embedding_1, tf.Tensor) and isinstance(embedding_2,tf.Tensor):
                embedding_1 = embedding_1.numpy()
                embedding_2 = embedding_2.numpy()
            similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
            series_result.append(similarity.item())
        except Exception as e:
            diagnostic_logger.warning(
                f"prompt_response_similarity encountered error {e} for text: {text}"
            )
            series_result.append(None)
    return series_result
