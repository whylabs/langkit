from typing import Optional
from whylogs.experimental.core.metrics.udf_metric import register_metric_udf
from . import LangKitConfig

lang_config = LangKitConfig()
_prompt = lang_config.prompt_column

_model_path = "JasperLS/gelectra-base-injection"
_tokenizer = None
_text_classification_pipeline = None


def init(model_path: Optional[str] = None):
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TextClassificationPipeline,
    )

    global _tokenizer, _text_classification_pipeline
    if model_path is None:
        model_path = _model_path
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    _text_classification_pipeline = TextClassificationPipeline(
        model=model, tokenizer=_tokenizer
    )


@register_metric_udf(col_name=_prompt)
def injection(text) -> float:
    if _text_classification_pipeline is None or _tokenizer is None:
        raise ValueError("Must initialize injections udf before evaluation.")
    result = _text_classification_pipeline(
        text, truncation=True, max_length=_tokenizer.model_max_length
    )
    injection_score = (
        result[0]["score"]
        if result[0]["label"] == "INJECTION"
        else 1 - result[0]["score"]
    )
    return injection_score


init()
