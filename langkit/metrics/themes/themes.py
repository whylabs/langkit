import json
import logging
import os
from functools import partial
from typing import Any, Dict, List, Optional, TypedDict, cast

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from langkit.core.metric import Metric, SingleMetric, SingleMetricResult, UdfInput
from langkit.metrics.embeddings_types import EmbeddingEncoder, TransformerEmbeddingAdapter
from langkit.metrics.embeddings_utils import compute_embedding_similarity

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


def load_themes(file_path: Optional[str] = None) -> Dict[str, List[str]]:
    __current_module_path = os.path.dirname(__file__)
    __default_pattern_file = file_path or os.path.join(__current_module_path, "themes.json")
    try:
        with open(__default_pattern_file, "r", encoding="utf-8") as f:
            themes_groups = json.loads(f.read())
            assert _validate_themes(themes_groups)
            return cast(Dict[str, List[str]], themes_groups)
    except FileNotFoundError as e:
        logger.error(f"Could not find {file_path}")
        raise e
    except json.decoder.JSONDecodeError as e:
        logger.warning(f"Could not parse {file_path}: {e}")
        raise e


def __themes_module(column_name: str, themes_group: str, embedding_encoder: Optional[EmbeddingEncoder] = None) -> Metric:
    # Load the themes into a Dictionary
    __default_themes_groups: Dict[str, List[str]] = load_themes()

    # Load the encoder
    if embedding_encoder is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = TransformerEmbeddingAdapter(SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device))
    else:
        encoder = embedding_encoder

    def __group_similarity(input_text: pd.DataFrame, input_column_name: str, group: str) -> float:
        input_to_list: List[str] = UdfInput(text=input_text).to_list(input_column_name)

        similarities = [
            compute_embedding_similarity(encoder, input_to_list, [themes_strings])
            for themes_strings in __default_themes_groups.get(group, [])
        ]
        max_similarities = [t.max() for t in similarities]  # type: ignore[reportUnknownMemeberType]

        return torch.max(torch.stack(max_similarities)).item() if max_similarities else 0.0  # type: ignore[reportUnknownMemeberType]

    def udf(text: pd.DataFrame) -> SingleMetricResult:
        metrics = [__group_similarity(text, column_name, themes_group)]
        return SingleMetricResult(metrics)

    return SingleMetric(
        name=f"{column_name}.{themes_group}_similarity",
        input_name=column_name,
        evaluate=udf,
    )


prompt_jailbreak_similarity_metric = partial(__themes_module, column_name="prompt", themes_group="jailbreak")
response_refusal_similarity_metric = partial(__themes_module, column_name="response", themes_group="refusal")
