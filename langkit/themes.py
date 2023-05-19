import json
from logging import getLogger
from typing import Optional

from sentence_transformers import util
from torch import Tensor
from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import register_metric_udf

from langkit.transformer import load_model

from . import LangKitConfig

diagnostic_logger = getLogger(__name__)

_transformer_model = None
_theme_groups = None

lang_config = LangKitConfig()


def register_theme_udfs():
    if "jailbreaks" in _theme_groups:
        jailbreak_embeddings = [
            _transformer_model.encode(s, convert_to_tensor=True)
            for s in _theme_groups["jailbreaks"]
        ]

        @register_metric_udf(col_type=String)
        def jailbreak_similarity(text: str) -> float:
            similarities = []
            for embedding in jailbreak_embeddings:
                similarity = get_subject_similarity(text, embedding)
                similarities.append(similarity)
            return max(similarities)

    if "refusals" in _theme_groups:
        refusal_embeddings = [
            _transformer_model.encode(s, convert_to_tensor=True)
            for s in _theme_groups["refusals"]
        ]

        @register_metric_udf(col_type=String)
        def refusal_similarity(text: str) -> float:
            similarities = []
            for embedding in refusal_embeddings:
                similarity = get_subject_similarity(text, embedding)
                similarities.append(similarity)
            return max(similarities)


def load_themes(json_path: str):
    try:
        skip = False
        with open(json_path, "r") as myfile:
            theme_groups = json.load(myfile)
    except FileNotFoundError:
        skip = True
        diagnostic_logger.warning(f"Could not find {json_path}")
    except json.decoder.JSONDecodeError as json_error:
        skip = True
        diagnostic_logger.warning(f"Could not parse {json_path}: {json_error}")
    if not skip:
        return theme_groups
    return None


def init(transformer_name: Optional[str] = None, theme_file_path: Optional[str] = None):
    global _transformer_model
    global _theme_groups
    if transformer_name is None:
        transformer_name = lang_config.transformer_name
    if theme_file_path is None:
        _theme_groups = load_themes(lang_config.theme_file_path)
    else:
        _theme_groups = load_themes(theme_file_path)

    _transformer_model = load_model(transformer_name)

    register_theme_udfs()


def get_subject_similarity(text: str, comparison_embedding: Tensor) -> float:
    if _transformer_model is None:
        raise ValueError("Must initialize a transformer before calling encode!")
    embedding = _transformer_model.encode(text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding, comparison_embedding)
    return similarity.item()


init()
