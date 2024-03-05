from copy import deepcopy
from typing import Optional
from functools import lru_cache
from whylogs.experimental.core.udf_schema import register_dataset_udf
from langkit import LangKitConfig, lang_config, prompt_column, response_column
import os
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)

_USE_CUDA = torch.cuda.is_available() and not bool(
    os.environ.get("LANGKIT_NO_CUDA", False)
)
_device = 0 if _USE_CUDA else -1

_prompt = prompt_column
_response = response_column

_model_path: Optional[str] = None


@lru_cache(maxsize=None)
def _get_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path)


@lru_cache(maxsize=None)
def _get_model(model_path: str):
    return AutoModelForSequenceClassification.from_pretrained(model_path)


@lru_cache(maxsize=None)
def _get_pipeline(model_path: str):
    return TextClassificationPipeline(
        model=_get_model(model_path),
        tokenizer=_get_tokenizer(model_path),
        device=_device,
    )


def toxicity(text: str) -> float:
    toxicity_pipeline = _get_pipeline(_model_path)
    toxicity_tokenizer = _get_tokenizer(_model_path)
    result = toxicity_pipeline(
        text, truncation=True, max_length=toxicity_tokenizer.model_max_length
    )
    return (
        result[0]["score"] if result[0]["label"] == "toxic" else 1 - result[0]["score"]
    )


def init_model():
    if _model_path is None:
        raise ValueError("Must initialize model path before calling toxicity!")
    _get_model(_model_path)
    _get_tokenizer(_model_path)
    _get_pipeline(_model_path)


@register_dataset_udf([_prompt], f"{_prompt}.toxicity")
def prompt_toxicity(text):
    init_model()
    return [toxicity(t) for t in text[_prompt]]


@register_dataset_udf([_response], f"{_response}.toxicity")
def response_toxicity(text):
    init_model()
    return [toxicity(t) for t in text[_response]]


def init(model_path: Optional[str] = None, config: Optional[LangKitConfig] = None):
    global _model_path
    config = config or deepcopy(lang_config)
    _model_path = model_path or config.toxicity_model_path


init()
