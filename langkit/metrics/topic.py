from dataclasses import dataclass
from functools import partial
from typing import List, Optional

import pandas as pd
import torch
from transformers import Pipeline, pipeline  # type: ignore

from langkit.core.metric import Metric, MetricCreator, SingleMetric, SingleMetricResult, UdfInput
from langkit.metrics.util import LazyInit

# TODO these default topics are not very good. We should update these to something reasonable that people would use.
__default_topics = ["politics", "economy", "entertainment", "environment"]

__classifier: LazyInit[Pipeline] = LazyInit(
    lambda: pipeline(
        "zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device="cuda" if torch.cuda.is_available() else "cpu"
    )
)


def __get_closest_topic(text: str, topics: List[str], multi_label: bool = False) -> Optional[str]:
    if not text:
        return None
    return __classifier.value(text, topics, multi_label=multi_label)["labels"][0]  # type: ignore


def topic_metric(column_name: str, topics: List[str]) -> Metric:
    def udf(text: pd.DataFrame) -> SingleMetricResult:
        metrics = [__get_closest_topic(it, topics) for it in UdfInput(text).iter_column_rows(column_name)]
        return SingleMetricResult(metrics)

    def init():
        __classifier.value

    return SingleMetric(
        name=f"{column_name}.closest_topic",
        input_name=column_name,
        evaluate=udf,
        init=init,
    )


prompt_topic_module = partial(topic_metric, "prompt", __default_topics)
response_topic_module = partial(topic_metric, "response", __default_topics)
prompt_response_topic_module = [prompt_topic_module, response_topic_module]


@dataclass
class CustomTopicModules:
    prompt_topic_module: MetricCreator
    response_topic_module: MetricCreator
    prompt_response_topic_module: MetricCreator


def get_custom_topic_modules(topics: List[str]) -> CustomTopicModules:
    prompt_topic_module = partial(topic_metric, "prompt", topics)
    response_topic_module = partial(topic_metric, "response", topics)
    return CustomTopicModules(
        prompt_topic_module=prompt_topic_module,
        response_topic_module=response_topic_module,
        prompt_response_topic_module=[prompt_topic_module, response_topic_module],
    )
