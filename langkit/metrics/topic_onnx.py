# pyright: reportUnknownMemberType=none
# pyright: reportUnknownVariableType=none
# pyright: reportUnknownLambdaType=none

from dataclasses import dataclass
from functools import lru_cache, partial
from typing import List, Optional, TypedDict

import pandas as pd
import torch
from optimum.modeling_base import PreTrainedModel
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, Pipeline, PreTrainedTokenizerBase, pipeline  # type: ignore

from langkit.core.metric import MetricCreator, MultiMetric, MultiMetricResult, UdfInput

__default_topics = [
    "medicine",
    "economy",
    "technology",
    "entertainment",
]

_hypothesis_template = "This example is about {}"


def _download_assets():
    ORTModelForSequenceClassification.from_pretrained(
        "MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33",
        subfolder="onnx",
        file_name="model.onnx",
        export=False,
    )


def _get_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33")


def _get_model() -> PreTrainedModel:
    return ORTModelForSequenceClassification.from_pretrained(
        "MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33",
        subfolder="onnx",
        file_name="model.onnx",
        export=False,
        local_files_only=True,
    )


@lru_cache
def _get_classifier() -> Pipeline:
    return pipeline(
        "zero-shot-classification",
        model=_get_model(),  # pyright: ignore[reportArgumentType]
        tokenizer=_get_tokenizer(),  # pyright: ignore[reportArgumentType]
        truncation=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


class ClassificationResults(TypedDict):
    sequence: str
    labels: List[str]
    scores: List[float]


def __get_scores_per_label(
    text: List[str], topics: List[str], hypothesis_template: str = _hypothesis_template, multi_label: bool = True
) -> List[ClassificationResults]:
    if not text:
        return []

    classifier = _get_classifier()
    result: List[ClassificationResults] = classifier(text, topics, hypothesis_template=hypothesis_template, multi_label=multi_label)  # type: ignore
    return result


def _sanitize_metric_name(topic: str) -> str:
    """
    sanitize a metric name created from a topic. Replace white space with underscores, etc.
    """
    return topic.replace(" ", "_").lower()


def topic_metric(input_name: str, topics: List[str], hypothesis_template: Optional[str] = None) -> MultiMetric:
    hypothesis_template = hypothesis_template or _hypothesis_template

    def udf(text: pd.DataFrame) -> MultiMetricResult:
        value: List[str] = list(UdfInput(text).iter_column_rows(input_name))
        results = __get_scores_per_label(value, topics=topics, hypothesis_template=hypothesis_template)

        all_metrics: List[List[float]] = [[] for _ in topics]
        for result in results:
            # Map each topic to its score in the current result
            topic_to_score = {label: score for label, score in zip(result["labels"], result["scores"])}
            # For each topic, append the score to the corresponding list in all_metrics
            for i, topic in enumerate(topics):
                all_metrics[i].append(topic_to_score[topic])  # Append list of score for the topic

        return MultiMetricResult(metrics=all_metrics)

    def cache_assets():
        _download_assets()

    def init():
        _get_classifier()

    metric_names = [f"{input_name}.topics.{_sanitize_metric_name(topic)}" for topic in topics]
    return MultiMetric(names=metric_names, input_names=[input_name], evaluate=udf, cache_assets=cache_assets, init=init)


prompt_topic_module = partial(topic_metric, "prompt", __default_topics, _hypothesis_template)
response_topic_module = partial(topic_metric, "response", __default_topics, _hypothesis_template)
prompt_response_topic_module = [prompt_topic_module, response_topic_module, _hypothesis_template]


@dataclass
class CustomTopicModules:
    prompt_topic_module: MetricCreator
    response_topic_module: MetricCreator
    prompt_response_topic_module: MetricCreator


def get_custom_topic_modules(topics: List[str], template: str = _hypothesis_template) -> CustomTopicModules:
    prompt_topic_module = partial(topic_metric, "prompt", topics, template)
    response_topic_module = partial(topic_metric, "response", topics, template)
    return CustomTopicModules(
        prompt_topic_module=prompt_topic_module,
        response_topic_module=response_topic_module,
        prompt_response_topic_module=[prompt_topic_module, response_topic_module],
    )
