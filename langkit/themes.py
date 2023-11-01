import json
from copy import deepcopy
from logging import getLogger
from typing import Callable, Optional, Dict, List, Set

from sentence_transformers import util
from torch import Tensor
from whylogs.experimental.core.udf_schema import register_dataset_udf

from langkit.transformer import Encoder

from . import LangKitConfig, lang_config, prompt_column, response_column
from langkit.whylogs.unreg import unregister_udfs


diagnostic_logger = getLogger(__name__)

_transformer_model = None
_theme_groups = None
_embeddings_map: Dict[str, List] = {}


_response_transformer_model = None
_response_theme_groups = None
_response_embeddings_map: Dict[str, List] = {}


def create_similarity_function(
    group: str, column: str, transformer_model, embeddings_map: Dict[str, List]
):
    def similarity_by_group(text):
        result = []
        for input in text[column]:
            score = group_similarity(input, group, transformer_model, embeddings_map)
            result.append(score)
        return result

    return similarity_by_group


def group_similarity(
    text: str, group, transformer_model, embeddings_map: Dict[str, List]
):
    similarities: List[float] = []
    if transformer_model is None:
        raise ValueError("Must initialize a transformer before calling encode!")

    text_embedding = transformer_model.encode(text)
    for embedding in embeddings_map.get(group, []):
        similarity = get_embeddings_similarity(text_embedding, embedding)
        similarities.append(similarity)
    return max(similarities) if similarities else None


def _map_embeddings(embeddings_map, theme_groups, transformer_model):
    for group in theme_groups:
        embeddings_map[group] = [
            transformer_model.encode(s) for s in theme_groups.get(group, [])
        ]


_registered: Set[str] = set()


def _register_theme_udfs():
    global _registered
    unregister_udfs(_registered)
    if _transformer_model is not None:
        _map_embeddings(_embeddings_map, _theme_groups, _transformer_model)
        for group in _theme_groups:
            column = prompt_column
            if group == "refusal":
                continue
            udf_name = f"{column}.{group}_similarity"
            _registered.add(udf_name)
            register_dataset_udf([column], udf_name=udf_name)(
                create_similarity_function(
                    group, column, _transformer_model, _embeddings_map
                )
            )

    if _response_transformer_model is not None:
        _map_embeddings(
            _response_embeddings_map,
            _response_theme_groups,
            _response_transformer_model,
        )
        for group in _response_theme_groups:
            column = response_column
            if group == "jailbreak":
                continue
            udf_name = f"{column}.{group}_similarity"
            _registered.add(udf_name)
            register_dataset_udf([column], udf_name=udf_name)(
                create_similarity_function(
                    group,
                    column,
                    _response_transformer_model,
                    _response_embeddings_map,
                )
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
    language: Optional[str] = None,
    transformer_name: Optional[str] = None,
    custom_encoder: Optional[Callable] = None,
    theme_file_path: Optional[str] = None,
    theme_json: Optional[str] = None,
    config: Optional[LangKitConfig] = None,
    response_transformer_name: Optional[str] = None,
    response_custom_encoder: Optional[Callable] = None,
    response_theme_file_path: Optional[str] = None,
    response_theme_json: Optional[str] = None,
):
    config = config or deepcopy(lang_config)
    global _transformer_model
    global _theme_groups
    if not transformer_name and not custom_encoder:
        transformer_name = config.transformer_name
    if not transformer_name and not custom_encoder:
        _transformer_model = None
    else:
        _transformer_model = Encoder(transformer_name, custom_encoder)

    if theme_file_path is not None and theme_json is not None:
        raise ValueError("Cannot specify both theme_file_path and theme_json")

    theme_file_path = theme_file_path or config.theme_file_path
    if theme_json:
        _theme_groups = json.loads(theme_json)
    elif theme_file_path:
        _theme_groups = load_themes(theme_file_path)
    else:
        _transformer_model = None

    global _response_transformer_model
    global _response_theme_groups
    if not response_transformer_name and not response_custom_encoder:
        response_transformer_name = config.response_transformer_name
    if not response_transformer_name and not response_custom_encoder:
        _response_transformer_model = None
    else:
        _response_transformer_model = Encoder(
            response_transformer_name, response_custom_encoder
        )
    if response_theme_file_path is not None and response_theme_json is not None:
        raise ValueError(
            "Cannot specify both response_theme_file_path and response_theme_json"
        )
    response_theme_file_path = (
        response_theme_file_path or config.response_theme_file_path
    )
    if response_theme_json:
        _response_theme_groups = json.loads(response_theme_json)
    elif response_theme_file_path:
        _response_theme_groups = load_themes(response_theme_file_path)
    else:
        _response_transformer_model = None

    _register_theme_udfs()


def get_subject_similarity(
    text: str, comparison_embedding: Tensor, transformer_model
) -> float:
    if transformer_model is None:
        raise ValueError("Must initialize a transformer before calling encode!")
    embedding = transformer_model.encode(text)
    similarity = util.pytorch_cos_sim(embedding, comparison_embedding)
    return similarity.item()


def get_embeddings_similarity(
    text_embedding: Tensor, comparison_embedding: Tensor
) -> float:
    similarity = util.pytorch_cos_sim(text_embedding, comparison_embedding)
    return similarity.item()
