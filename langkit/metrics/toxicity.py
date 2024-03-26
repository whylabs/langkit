# pyright: reportUnknownMemberType=none
# pyright: reportUnknownVariableType=none
# pyright: reportUnknownLambdaType=none
import os
from functools import lru_cache, partial
from typing import List, cast

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TextClassificationPipeline,
)

from langkit.core.metric import Metric, SingleMetric, SingleMetricResult, UdfInput


def __toxicity(pipeline: TextClassificationPipeline, max_length: int, text: List[str]) -> List[float]:
    results = pipeline(text, truncation=True, max_length=max_length)
    return [result["score"] if result["label"] == "toxic" else 1.0 - result["score"] for result in results]  # type: ignore


__model_path = "martin-ha/toxic-comment-model"


@lru_cache
def _get_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(__model_path)


@lru_cache
def _get_pipeline() -> TextClassificationPipeline:
    use_cuda = torch.cuda.is_available() and not bool(os.environ.get("LANGKIT_NO_CUDA", False))
    model: PreTrainedTokenizerBase = AutoModelForSequenceClassification.from_pretrained(__model_path)
    tokenizer = _get_tokenizer()
    return TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0 if use_cuda else -1)


def toxicity_metric(column_name: str) -> Metric:
    def init():
        _get_pipeline()

    def udf(text: pd.DataFrame) -> SingleMetricResult:
        _tokenizer = _get_tokenizer()
        _pipeline = _get_pipeline()

        col = list(UdfInput(text).iter_column_rows(column_name))
        max_length = cast(int, _tokenizer.model_max_length)
        metrics = __toxicity(_pipeline, max_length, col)
        return SingleMetricResult(metrics=metrics)

    return SingleMetric(name=f"{column_name}.toxicity.toxicity_score", input_names=[column_name], evaluate=udf, init=init)


prompt_toxicity_metric = partial(toxicity_metric, "prompt")
response_toxicity_metric = partial(toxicity_metric, "response")
prompt_response_toxicity_module = [prompt_toxicity_metric, response_toxicity_metric]
