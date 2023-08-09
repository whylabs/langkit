import json
from logging import getLogger
from typing import Optional, Dict, List

from sentence_transformers import util
from torch import Tensor
from whylogs.experimental.core.udf_schema import register_dataset_udf

from langkit.transformer import load_model

from . import LangKitConfig

diagnostic_logger = getLogger(__name__)

_transformer_model = None
_theme_groups = None
lang_config = LangKitConfig()
_prompt = lang_config.prompt_column
_response = lang_config.response_column

_embeddings_map: Dict[str, List] = {}


def create_similarity_function(group: str, column: str):
    def similarity_by_group(text):
        result = []
        for input in text[column]:
            score = group_similarity(input, group)
            result.append(score)
        return result

    return similarity_by_group


def group_similarity(text: str, group):
    similarities: List[float] = []
    if _transformer_model is None:
        raise ValueError("Must initialize a transformer before calling encode!")

    text_embedding = _transformer_model.encode(text, convert_to_tensor=True)
    for embedding in _embeddings_map.get(group, []):
        similarity = get_embeddings_similarity(text_embedding, embedding)
        similarities.append(similarity)
    return max(similarities) if similarities else None


def _map_embeddings():
    global _embeddings_map
    for group in _theme_groups:
        _embeddings_map[group] = [
            _transformer_model.encode(s, convert_to_tensor=True)
            for s in _theme_groups.get(group, [])
        ]


def register_theme_udfs():
    _map_embeddings()

    for group in _theme_groups:
        for column in [_prompt, _response]:
            if group == "jailbreak" and column == _response:
                continue
            if group == "refusal" and column == _prompt:
                continue
            register_dataset_udf([column], udf_name=f"{column}.{group}_similarity")(
                create_similarity_function(group, column)
            )


def load_themes(json_path: str, encoding="utf-8"):
    try:
        skip = False
        with open(json_path, "r", encoding=encoding) as myfile:
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


def init(
    transformer_name: Optional[str] = None,
    theme_file_path: Optional[str] = None,
    theme_json: Optional[str] = None,
):
    global _transformer_model
    global _theme_groups
    if transformer_name is None:
        transformer_name = lang_config.transformer_name
    if theme_file_path is not None and theme_json is not None:
        raise ValueError("Cannot specify both theme_file_path and theme_json")
    if theme_file_path is None:
        if theme_json:
            _theme_groups = json.loads(theme_json)
        else:
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


def get_embeddings_similarity(
    text_embedding: Tensor, comparison_embedding: Tensor
) -> float:
    if _transformer_model is None:
        raise ValueError("Must initialize a transformer before calling encode!")
    similarity = util.pytorch_cos_sim(text_embedding, comparison_embedding)
    return similarity.item()


init()
