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


_model_path = "martin-ha/toxic-comment-model"
_revision = "9842c08b35a4687e7b211187d676986c8c96256d"


def _cache_assets():
    AutoModelForSequenceClassification.from_pretrained(_model_path, revision=_revision)
    AutoTokenizer.from_pretrained(_model_path, revision=_revision)


@lru_cache
def _get_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(_model_path, local_files_only=True, revision=_revision)


@lru_cache
def _get_pipeline() -> TextClassificationPipeline:
    use_cuda = torch.cuda.is_available() and not bool(os.environ.get("LANGKIT_NO_CUDA", False))
    model: PreTrainedTokenizerBase = AutoModelForSequenceClassification.from_pretrained(
        _model_path, local_files_only=True, revision=_revision
    )
    tokenizer = _get_tokenizer()
    return TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0 if use_cuda else -1)


def toxicity_metric(column_name: str) -> Metric:
    def cache_assets():
        _cache_assets()

    def init():
        _get_pipeline()

    def udf(text: pd.DataFrame) -> SingleMetricResult:
        _tokenizer = _get_tokenizer()
        _pipeline = _get_pipeline()

        col = list(UdfInput(text).iter_column_rows(column_name))
        max_length = cast(int, _tokenizer.model_max_length)
        metrics = __toxicity(_pipeline, max_length, col)
        return SingleMetricResult(metrics=metrics)

    return SingleMetric(
        name=f"{column_name}.toxicity.toxicity_score", input_names=[column_name], evaluate=udf, init=init, cache_assets=cache_assets
    )


prompt_toxicity_metric = partial(toxicity_metric, "prompt")
response_toxicity_metric = partial(toxicity_metric, "response")
prompt_response_toxicity_module = [prompt_toxicity_metric, response_toxicity_metric]
