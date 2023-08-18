from logging import getLogger
from typing import Callable, Optional

from sentence_transformers import util
from whylogs.experimental.core.udf_schema import register_dataset_udf
from . import lang_config, prompt_column, response_column
from langkit.transformer import Encoder

_prompt = prompt_column
_response = response_column


_transformer_model = None

diagnostic_logger = getLogger(__name__)


def init(
    transformer_name: Optional[str] = None, custom_encoder: Optional[Callable] = None
):
    global _transformer_model
    if transformer_name is None and custom_encoder is None:
        transformer_name = lang_config.transformer_name
    _transformer_model = Encoder(transformer_name, custom_encoder)


init()


@register_dataset_udf([_prompt, _response], f"{_response}.relevance_to_{_prompt}")
def prompt_response_similarity(text):
    global _transformer_model

    if _transformer_model is None:
        raise ValueError(
            "response.relevance_to_prompt must have a transformer model initialized before use."
        )

    series_result = []
    for x, y in zip(text["prompt"], text["response"]):
        try:
            embedding_1 = _transformer_model.encode([x] if isinstance(x, str) else x)
            embedding_2 = _transformer_model.encode([y] if isinstance(y, str) else y)
            similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
            series_result.append(similarity.item())
        except Exception as e:
            diagnostic_logger.warning(
                f"prompt_response_similarity encountered error {e} for text: {text}"
            )
            series_result.append(None)
    return series_result
