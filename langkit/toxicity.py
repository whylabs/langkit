from copy import deepcopy
from typing import Optional

from whylogs.experimental.core.udf_schema import register_dataset_udf
from . import LangKitConfig, lang_config, prompt_column, response_column


_toxicity_tokenizer = None
_toxicity_pipeline = None

_response_toxicity_tokenizer = None
_response_toxicity_pipeline = None


def toxicity(text: str, pipeline, tokenizer) -> float:
    if pipeline is None or tokenizer is None:
        raise ValueError("toxicity score must initialize the pipeline first")

    result = pipeline(text, truncation=True, max_length=tokenizer.model_max_length)
    return (
        result[0]["score"] if result[0]["label"] == "toxic" else 1 - result[0]["score"]
    )


def _toxicity_wrapper(column, pipeline, tokenizer):
    return lambda text: [toxicity(t, pipeline, tokenizer) for t in text[column]]


def init(
    language: Optional[str] = None,
    model_path: Optional[str] = None,
    config: Optional[LangKitConfig] = None,
    response_model_path: Optional[str] = None,
):
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TextClassificationPipeline,
    )

    config = config or deepcopy(lang_config)
    model_path = model_path or config.toxicity_model_path
    global _toxicity_tokenizer, _toxicity_pipeline
    if model_path is None:
        _toxicity_pipeline = None
    else:
        _toxicity_tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        _toxicity_pipeline = TextClassificationPipeline(
            model=model, tokenizer=_toxicity_tokenizer
        )
        register_dataset_udf([prompt_column], f"{prompt_column}.toxicity")(
            _toxicity_wrapper(prompt_column, _toxicity_pipeline, _toxicity_tokenizer)
        )

    model_path = response_model_path or config.response_toxicity_model_path
    global _response_toxicity_tokenizer, _response_toxicity_pipeline
    if model_path is None:
        _response_toxicity_pipeline = None
    else:
        _response_toxicity_tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        _response_toxicity_pipeline = TextClassificationPipeline(
            model=model, tokenizer=_response_toxicity_tokenizer
        )
        register_dataset_udf([response_column], f"{response_column}.toxicity")(
            _toxicity_wrapper(
                response_column,
                _response_toxicity_pipeline,
                _response_toxicity_tokenizer,
            )
        )
