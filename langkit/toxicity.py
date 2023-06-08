from typing import Optional

from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import register_metric_udf

_toxicity_model_path = "martin-ha/toxic-comment-model"
_toxicity_tokenizer = None
_toxicity_pipeline = None


def init(model_path: Optional[str] = None):
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TextClassificationPipeline,
    )

    global _toxicity_tokenizer, _toxicity_pipeline
    if model_path is None:
        model_path = _toxicity_model_path
    _toxicity_tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    _toxicity_pipeline = TextClassificationPipeline(
        model=model, tokenizer=_toxicity_tokenizer
    )


@register_metric_udf(col_type=String)
def toxicity(text: str) -> float:
    if _toxicity_pipeline is None or _toxicity_tokenizer is None:
        raise ValueError("toxicity score must initialize the pipeline first")
    result = _toxicity_pipeline(
        text, truncation=True, max_length=_toxicity_tokenizer.model_max_length
    )
    toxicity_score = (
        result[0]["score"] if result[0]["label"] == "toxic" else 1 - result[0]["score"]
    )
    return toxicity_score


init()
