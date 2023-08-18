import json
from logging import getLogger
from typing import Callable, Optional, Dict, List

from sentence_transformers import util
from torch import Tensor
from whylogs.experimental.core.udf_schema import register_dataset_udf

from langkit.transformer import Encoder

from . import lang_config, prompt_column, response_column

diagnostic_logger = getLogger(__name__)

_transformer_model = None
_theme_groups = None
_prompt = prompt_column
_response = response_column

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

    text_embedding = _transformer_model.encode(text)
    for embedding in _embeddings_map.get(group, []):
        similarity = get_embeddings_similarity(text_embedding, embedding)
        similarities.append(similarity)
    return max(similarities) if similarities else None


def _map_embeddings():
    global _embeddings_map
    for group in _theme_groups:
        _embeddings_map[group] = [
            _transformer_model.encode(s) for s in _theme_groups.get(group, [])
        ]


_registered = set()


def _register_theme_udfs():
    global _registered
    _map_embeddings()

    for group in _theme_groups:
        for column in [_prompt, _response]:
            if group == "jailbreak" and column == _response:
                continue
            if group == "refusal" and column == _prompt:
                continue
            udf_name = f"{column}.{group}_similarity"
            if udf_name not in _registered:
                _registered.add(udf_name)
                register_dataset_udf([column], udf_name=udf_name)(
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
    custom_encoder: Optional[Callable] = None,
    theme_file_path: Optional[str] = None,
    theme_json: Optional[str] = None,
):
    global _transformer_model
    global _theme_groups
    if not transformer_name and not custom_encoder:
        transformer_name = lang_config.transformer_name
    _transformer_model = Encoder(transformer_name, custom_encoder)
    if theme_file_path is not None and theme_json is not None:
        raise ValueError("Cannot specify both theme_file_path and theme_json")
    if theme_file_path is None:
        if theme_json:
            _theme_groups = json.loads(theme_json)
        else:
            _theme_groups = load_themes(lang_config.theme_file_path)
    else:
        _theme_groups = load_themes(theme_file_path)
    _register_theme_udfs()


def get_subject_similarity(text: str, comparison_embedding: Tensor) -> float:
    if _transformer_model is None:
        raise ValueError("Must initialize a transformer before calling encode!")
    embedding = _transformer_model.encode(text)
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
