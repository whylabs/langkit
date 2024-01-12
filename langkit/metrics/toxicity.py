# pyright: reportUnknownMemberType=none
# pyright: reportUnknownVariableType=none
import os
from functools import partial
from typing import List, Optional, cast

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TextClassificationPipeline,
)

from langkit.metrics.metric import Metric, MetricResult, UdfInput


def __toxicity(pipeline: TextClassificationPipeline, max_length: int, text: List[str]) -> List[float]:
    # TODO lots of error handling here
    results = pipeline(text, truncation=True, max_length=max_length)
    return [result["score"] if result["label"] == "toxic" else 1.0 - result["score"] for result in results]  # type: ignore


__model: Optional[PreTrainedTokenizerBase] = None
__tokenizer: Optional[PreTrainedTokenizerBase] = None


def __toxicity_module(column_name: str) -> Metric:
    global __model, __tokenizer
    model_path = "martin-ha/toxic-comment-model"

    if __model is None:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        __model = model
    else:
        model = __model

    if __tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        __tokenizer = tokenizer
    else:
        tokenizer = __tokenizer

    use_cuda = torch.cuda.is_available() and not bool(os.environ.get("LANGKIT_NO_CUDA", False))
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0 if use_cuda else -1)  # type: ignore[reportUnknownArgumentType]
    # TODO test/error handling
    max_length = cast(int, tokenizer.model_max_length)

    def udf(text: pd.DataFrame) -> MetricResult:
        col = list(UdfInput(text).iter_column_rows(column_name))
        metrics = __toxicity(pipeline, max_length, col)
        return MetricResult(metrics=metrics)

    return Metric(name=f"{column_name}.toxicity", input_name=column_name, evaluate=udf)


prompt_toxicity_module = partial(__toxicity_module, "prompt")
response_toxicity_module = partial(__toxicity_module, "response")
# TODO this has to be a readonly list
prompt_response_toxicity_module = [prompt_toxicity_module, response_toxicity_module]
