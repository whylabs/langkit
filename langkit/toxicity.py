from copy import deepcopy
from typing import Optional

from whylogs.experimental.core.udf_schema import register_dataset_udf
from langkit import LangKitConfig, lang_config, prompt_column, response_column
from langkit.utils import LazyInit
import os
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TextClassificationPipeline,
)

_USE_CUDA = torch.cuda.is_available() and not bool(
    os.environ.get("LANGKIT_NO_CUDA", False)
)
_device = 0 if _USE_CUDA else -1

_prompt = prompt_column
_response = response_column
_toxicity_tokenizer = None
_toxicity_pipeline = None

_model_path: Optional[str] = None

_model = LazyInit(
    lambda : AutoModelForSequenceClassification.from_pretrained(_model_path)
)
_toxicity_tokenizer = LazyInit(lambda: AutoTokenizer.from_pretrained(_model_path))
__use_cuda = torch.cuda.is_available() and not bool(os.environ.get("LANGKIT_NO_CUDA", False))
_toxicity_pipeline = LazyInit(
    lambda: TextClassificationPipeline(
        model=_model.value, tokenizer=_toxicity_tokenizer.value, device=0 if __use_cuda else -1
    )
)


def toxicity(text: str) -> float:
    toxicity_pipeline = _toxicity_pipeline.value
    toxicity_tokenizer = _toxicity_tokenizer.value
    result = toxicity_pipeline(
        text, truncation=True, max_length=toxicity_tokenizer.model_max_length
    )
    return (
        result[0]["score"] if result[0]["label"] == "toxic" else 1 - result[0]["score"]
    )


def init_model():
    if _model_path is None:
        raise ValueError("Must initialize model path before calling toxicity!")
    _model.value
    _toxicity_tokenizer.value
    _toxicity_pipeline.value


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
