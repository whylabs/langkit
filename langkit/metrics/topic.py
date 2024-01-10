from dataclasses import dataclass
from functools import partial
from typing import List

import pandas as pd
import torch
from transformers import Pipeline, pipeline  # type: ignore

from langkit.metrics.metric import EvaluationResult, Metric, MetricConfig, UdfInput
from langkit.metrics.util import LazyInit

__default_topics = ["politics", "economy", "entertainment", "environment"]

__classifier: LazyInit[Pipeline] = LazyInit(
    lambda: pipeline(
        "zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device="cuda" if torch.cuda.is_available() else "cpu"
    )
)


def __get_closest_topic(text: str, topics: List[str], multi_label: bool = False) -> str:
    return __classifier.value(text, topics, multi_label=multi_label)["labels"][0]  # type: ignore


def __topic_module(column_name: str, topics: List[str]) -> MetricConfig:
    def udf(text: pd.DataFrame) -> EvaluationResult:
        metrics = [__get_closest_topic(it, topics) for it in UdfInput(text).iter_column_rows(column_name)]
        return EvaluationResult(metrics)

    return MetricConfig(
        name=f"{column_name}.closest_topic",
        input_name=column_name,
        evaluate=udf,
    )


prompt_topic_module = partial(__topic_module, "prompt", __default_topics)
response_topic_module = partial(__topic_module, "response", __default_topics)
prompt_response_topic_module = [prompt_topic_module, response_topic_module]


@dataclass
class CustomTopicModules:
    prompt_topic_module: Metric
    response_topic_module: Metric
    prompt_response_topic_module: Metric


def get_custom_topic_modules(topics: List[str]) -> CustomTopicModules:
    prompt_topic_module = partial(__topic_module, "prompt", topics)
    response_topic_module = partial(__topic_module, "response", topics)
    return CustomTopicModules(
        prompt_topic_module=prompt_topic_module,
        response_topic_module=response_topic_module,
        prompt_response_topic_module=[prompt_topic_module, response_topic_module],
    )
