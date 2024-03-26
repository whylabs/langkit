import json
import logging
import os
from functools import lru_cache, partial
from typing import Any, Dict, List, Literal, TypedDict, cast

import pandas as pd
import torch
import torch.nn.functional as F

from langkit.core.metric import Metric, SingleMetric, SingleMetricResult
from langkit.metrics.embeddings_types import TransformerEmbeddingAdapter
from langkit.transformer import embedding_adapter

logger = logging.getLogger(__name__)


class Themes(TypedDict):
    key: Dict[str, List[str]]


def _validate_themes(data: Any) -> bool:
    if not isinstance(data, dict):
        return False

    for key, value in data.items():  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(key, str):
            return False

        if not isinstance(value, list):
            return False

        # Check if all items in the list are strings
        if not all(isinstance(item, str) for item in value):  # pyright: ignore[reportUnknownVariableType]
            return False

    return True


def __load_themes() -> Dict[str, List[str]]:
    __current_module_path = os.path.dirname(__file__)
    __default_pattern_file = os.path.join(__current_module_path, "themes.json")
    try:
        with open(__default_pattern_file, "r", encoding="utf-8") as f:
            themes_groups = json.loads(f.read())
            assert _validate_themes(themes_groups)
            return cast(Dict[str, List[str]], themes_groups)
    except FileNotFoundError as e:
        logger.error(f"Could not find {__default_pattern_file}")
        raise e
    except json.decoder.JSONDecodeError as e:
        logger.error(f"Could not parse {__default_pattern_file}: {e}")
        raise e


@lru_cache
def _get_themes(encoder: TransformerEmbeddingAdapter) -> Dict[str, torch.Tensor]:
    theme_groups = __load_themes()
    return {group: torch.as_tensor(encoder.encode(tuple(themes))) for group, themes in theme_groups.items()}


def __themes_metric(column_name: str, themes_group: Literal["jailbreak", "refusal"]) -> Metric:
    if themes_group == "refusal" and column_name == "prompt":
        raise ValueError("Refusal themes are not applicable to prompt")

    if themes_group == "jailbreak" and column_name == "response":
        raise ValueError("Jailbreak themes are not applicable to response")

    def cache_assets():
        _get_themes(embedding_adapter())

    def udf(text: pd.DataFrame) -> SingleMetricResult:
        encoder = embedding_adapter()
        theme = _get_themes(encoder)[themes_group]  # (n_theme_examples, embedding_dim)
        text_list: List[str] = text[column_name].tolist()
        encoded_text = encoder.encode(tuple(text_list))  # (n_input_rows, embedding_dim)
        similarities = F.cosine_similarity(encoded_text.unsqueeze(1), theme.unsqueeze(0), dim=2)  # (n_input_rows, n_theme_examples)
        max_similarities = similarities.max(dim=1)[0]  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]  (n_input_rows,)
        similarity_list: List[float] = max_similarities.tolist()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportUnknownVariableType]
        return SingleMetricResult(similarity_list)  # pyright: ignore[reportUnknownArgumentType]

    return SingleMetric(
        name=f"{column_name}.similarity.{themes_group}",
        input_names=[column_name],
        evaluate=udf,
        cache_assets=cache_assets,
    )


prompt_jailbreak_similarity_metric = partial(__themes_metric, column_name="prompt", themes_group="jailbreak")
response_refusal_similarity_metric = partial(__themes_metric, column_name="response", themes_group="refusal")
