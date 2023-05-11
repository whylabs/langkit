from . import LangKitConfig
from whylogs.experimental.core.metrics.udf_metric import (
    register_metric_udf,
)
from typing import Optional
from sentence_transformers import SentenceTransformer, util
from torch import Tensor
from typing import Callable
from whylogs.core.datatypes import DataType
from whylogs.experimental.core.metrics.udf_metric import _col_type_submetrics
from logging import getLogger
import json
from langkit.transformer import load_model

diagnostic_logger = getLogger(__name__)

_transformer_model = None
_exclusion_groups = None

lang_config = LangKitConfig()
input_col_name = None
output_col_name = None

def register_exclusion_udfs():
    if "jailbreaks" in _exclusion_groups:
        jailbreak_embeddings = [
            _transformer_model.encode(s, convert_to_tensor=True)
            for s in _exclusion_groups["jailbreaks"]
        ]
        @register_metric_udf(col_name=input_col_name)
        def jailbreak_similarity(text: str) -> float:
            similarities = []
            for embedding in jailbreak_embeddings:
                similarity = get_subject_similarity(text, embedding)
                similarities.append(similarity)
            return max(similarities)

    if "refusals" in _exclusion_groups:
        refusal_embeddings = [
        _transformer_model.encode(s, convert_to_tensor=True)
        for s in _exclusion_groups["refusals"]
        ]
        @register_metric_udf(col_name=output_col_name)
        def refusal_similarity(text: str) -> float:
            similarities = []
            for embedding in refusal_embeddings:
                similarity = get_subject_similarity(text, embedding)
                similarities.append(similarity)
            return max(similarities)


def load_exclusions(json_path: str):
    try:
        skip = False
        with open(json_path, "r") as myfile:
            exclusion_groups = json.load(myfile)
    except FileNotFoundError:
        skip = True
        diagnostic_logger.warning(f"Could not find {json_path}")
    except json.decoder.JSONDecodeError as json_error:
        skip = True
        diagnostic_logger.warning(f"Could not parse {json_path}: {json_error}")
    if not skip:
        return exclusion_groups
    return None


def init(transformer_name: Optional[str]=None, input_col: Optional[str]=None, output_col: Optional[str]=None, exclusion_file_path: Optional[str]=None):
    global _transformer_model
    global _exclusion_groups
    global input_col_name
    global output_col_name
    if transformer_name is None:
        transformer_name = lang_config.transformer_name
    if input_col is None:
        input_col_name = lang_config.input_name
    else:
        input_col_name = input_col
    if output_col is None:
        output_col_name = lang_config.output_name
    else:
        output_col_name = output_col
    if exclusion_file_path is None:
        _exclusion_groups = load_exclusions(lang_config.exclusion_file_path)
    else:
        _exclusion_groups = load_exclusions(exclusion_file_path)

    _transformer_model = load_model(transformer_name)

    register_exclusion_udfs()


def get_subject_similarity(text: str, comparison_embedding: Tensor) -> float:
    embedding = _transformer_model.encode(text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding, comparison_embedding)
    return similarity.item()

init()
