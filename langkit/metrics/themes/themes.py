import json
import logging
import os
from functools import partial
from typing import Any, Dict, List, Literal, Optional, TypedDict, cast

import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from langkit.core.metric import Metric, SingleMetric, SingleMetricResult
from langkit.metrics.embeddings_types import EmbeddingEncoder, TransformerEmbeddingAdapter
from langkit.metrics.util import DynamicLazyInit

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


def __compute_theme_embeddings(theme_groups: Dict[str, List[str]], encoder: EmbeddingEncoder) -> Dict[str, torch.Tensor]:
    return {group: torch.as_tensor(encoder.encode(themes)) for group, themes in theme_groups.items()}


__themes = DynamicLazyInit[EmbeddingEncoder, Dict[str, torch.Tensor]](lambda encoder: __compute_theme_embeddings(__load_themes(), encoder))


def __themes_metric(
    column_name: str, themes_group: Literal["jailbreak", "refusal"], embedding_encoder: Optional[EmbeddingEncoder] = None
) -> Metric:
    if embedding_encoder is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = TransformerEmbeddingAdapter(SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device))
    else:
        encoder = embedding_encoder

    theme = __themes.value(encoder)[themes_group]  # (n_theme_examples, embedding_dim)

    def udf(text: pd.DataFrame) -> SingleMetricResult:
        text_list: List[str] = text[column_name].tolist()
        encoded_text = encoder.encode(text_list)  # (n_input_rows, embedding_dim)
        similarities = F.cosine_similarity(encoded_text.unsqueeze(1), theme.unsqueeze(0), dim=2)  # (n_input_rows, n_theme_examples)
        max_similarities = similarities.max(dim=1)[0]  # (n_input_rows,)
        similarity_list: List[float] = max_similarities.tolist()  # pyright: ignore[reportUnknownMemberType]
        return SingleMetricResult(similarity_list)

    return SingleMetric(
        name=f"{column_name}.{themes_group}_similarity",
        input_name=column_name,
        evaluate=udf,
    )


prompt_jailbreak_similarity_metric = partial(__themes_metric, column_name="prompt", themes_group="jailbreak")
response_refusal_similarity_metric = partial(__themes_metric, column_name="response", themes_group="refusal")
