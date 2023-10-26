from copy import deepcopy
from typing import Optional

from whylogs.experimental.core.udf_schema import register_dataset_udf
from . import LangKitConfig, lang_config, prompt_column, response_column


_prompt = prompt_column
_response = response_column
_toxicity_tokenizer = None
_toxicity_pipeline = None


def toxicity(text: str) -> float:
    if _toxicity_pipeline is None or _toxicity_tokenizer is None:
        raise ValueError("toxicity score must initialize the pipeline first")

    result = _toxicity_pipeline(
        text, truncation=True, max_length=_toxicity_tokenizer.model_max_length
    )
    return (
        result[0]["score"] if result[0]["label"] == "toxic" else 1 - result[0]["score"]
    )


def prompt_toxicity(text):
    return [toxicity(t) for t in text[_prompt]]


def response_toxicity(text):
    return [toxicity(t) for t in text[_response]]


def init(language: str = "en", model_path: Optional[str] = None, config: Optional[LangKitConfig] = None):
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TextClassificationPipeline,
    )

    config = config or deepcopy(lang_config)
    model_path = model_path or config.toxicity_model_path
    global _toxicity_tokenizer, _toxicity_pipeline
    _toxicity_tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    _toxicity_pipeline = TextClassificationPipeline(
        model=model, tokenizer=_toxicity_tokenizer
    )
    register_dataset_udf(
        [_prompt],
        f"{language}.{_prompt}.toxicity",
        schema_name=language
    )(prompt_toxicity)
    register_dataset_udf(
        [_response],
        f"{language}.{_response}.toxicity"
        schema_name=language
    )(response_toxicity)


init()
