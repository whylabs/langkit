from typing import Dict, List, Optional, Union
from logging import getLogger

from whylogs.core.datatypes import String
from whylogs.core.stubs import pd
from whylogs.experimental.core.udf_schema import register_type_udf

_toxicity_model_path = "martin-ha/toxic-comment-model"
_toxicity_tokenizer = None
_toxicity_pipeline = None

diagnostic_logger = getLogger(__name__)


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


@register_type_udf(String, namespace="sentiment")
def toxicity(strings) -> list:
    if _toxicity_pipeline is None or _toxicity_tokenizer is None:
        raise ValueError("toxicity score must initialize the pipeline first")
    series_result = []

    for text in strings:
        try:
            result = _toxicity_pipeline(
                text, truncation=True, max_length=_toxicity_tokenizer.model_max_length
            )
            toxicity_score = (
                result[0]["score"] if result[0]["label"] == "toxic" else 1 - result[0]["score"]
            )
            series_result.append(toxicity_score)
        except Exception as e:
            diagnostic_logger.warning(
                f"Exception in toxicity udf: {e}"
            )
            series_result.append(None)

    return series_result
    

init()
