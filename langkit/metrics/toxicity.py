# pyright: reportUnknownMemberType=none
# pyright: reportUnknownVariableType=none
# pyright: reportUnknownLambdaType=none
import os
from functools import partial
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
from langkit.metrics.util import DynamicLazyInit


def __toxicity(pipeline: TextClassificationPipeline, max_length: int, text: List[str]) -> List[float]:
    results = pipeline(text, truncation=True, max_length=max_length)
    return [result["score"] if result["label"] == "toxic" else 1.0 - result["score"] for result in results]  # type: ignore


__model: DynamicLazyInit[str, PreTrainedTokenizerBase] = DynamicLazyInit(
    lambda model_path: AutoModelForSequenceClassification.from_pretrained(model_path)
)
__tokenizer: DynamicLazyInit[str, PreTrainedTokenizerBase] = DynamicLazyInit(lambda model_path: AutoTokenizer.from_pretrained(model_path))
__use_cuda = torch.cuda.is_available() and not bool(os.environ.get("LANGKIT_NO_CUDA", False))
__pipeline: DynamicLazyInit[str, TextClassificationPipeline] = DynamicLazyInit(
    lambda model_path: TextClassificationPipeline(
        model=__model.value(model_path), tokenizer=__tokenizer.value(model_path), device=0 if __use_cuda else -1
    )
)


def toxicity_metric(column_name: str, model_path="martin-ha/toxic-comment-model") -> Metric:
    def cache_assets():
        __model.value(model_path)
        __tokenizer.value(model_path)
        __pipeline.value(model_path)

    def udf(text: pd.DataFrame) -> SingleMetricResult:
        _tokenizer = __tokenizer.value(model_path)
        _pipeline = __pipeline.value(model_path)

        col = list(UdfInput(text).iter_column_rows(column_name))
        max_length = cast(int, _tokenizer.model_max_length)
        metrics = __toxicity(_pipeline, max_length, col)
        return SingleMetricResult(metrics=metrics)

    return SingleMetric(name=f"{column_name}.toxicity", input_name=column_name, evaluate=udf, cache_assets=cache_assets)


prompt_toxicity_module = partial(toxicity_metric, "prompt")
response_toxicity_module = partial(toxicity_metric, "response")
prompt_response_toxicity_module = [prompt_toxicity_module, response_toxicity_module]
