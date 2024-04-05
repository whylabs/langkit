# pyright: reportUnknownMemberType=none
# pyright: reportUnknownVariableType=none
# pyright: reportUnknownLambdaType=none

from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from optimum.modeling_base import PreTrainedModel
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, Pipeline, PreTrainedTokenizerBase, pipeline  # type: ignore

from langkit.core.metric import MetricCreator, MultiMetric, MultiMetricResult

__default_topics = [
    "medicine",
    "economy",
    "technology",
    "entertainment",
]

_hypothesis_template = "This example is about {}"

__local_model_path = "xtremedistil-l6-h256-zeroshot-v1.1-all-33-onnx"
__onnx_model_path = "model_quantized.onnx"


@lru_cache
def _get_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(__local_model_path)


@lru_cache
def _get_model() -> PreTrainedModel:
    return ORTModelForSequenceClassification.from_pretrained(__local_model_path, file_name=__onnx_model_path, export=False)


@lru_cache
def _get_classifier() -> Pipeline:
    return pipeline(
        "zero-shot-classification",
        model=_get_model(),  # type: ignore
        tokenizer=_get_tokenizer(),  # type: ignore
        truncation=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def __get_scores_per_label(
    text: str, topics: List[str], hypothesis_template: str = _hypothesis_template, multi_label: bool = True
) -> Optional[Dict[str, float]]:
    if not text:
        return None
    classifier = _get_classifier()
    result: Dict[str, [str, float]] = classifier(text, topics, hypothesis_template=hypothesis_template, multi_label=multi_label)  # type: ignore
    scores_per_label: Dict[str, float] = {label: score for label, score in zip(result["labels"], result["scores"])}  # type: ignore[reportUnknownVariableType]
    return scores_per_label


def _sanitize_metric_name(topic: str) -> str:
    """
    sanitize a metric name created from a topic. Replace white space with underscores, etc.
    """
    return topic.replace(" ", "_").lower()


def topic_metric(input_name: str, topics: List[str], hypothesis_template: Optional[str] = None) -> MultiMetric:
    hypothesis_template = hypothesis_template or _hypothesis_template

    def udf(text: pd.DataFrame) -> MultiMetricResult:
        metrics: Dict[str, List[Optional[float]]] = {topic: [] for topic in topics}

        def process_row(row: pd.DataFrame) -> Dict[str, List[Optional[float]]]:
            value: Any = row[input_name]  # type: ignore
            scores = __get_scores_per_label(value, topics=topics, hypothesis_template=hypothesis_template)  # pyright: ignore[reportUnknownArgumentType]
            for topic in topics:
                metrics[topic].append(scores[topic] if scores else None)
            return metrics

        text.apply(process_row, axis=1)  # pyright: ignore[reportUnknownMemberType]

        all_metrics = [
            *metrics.values(),
        ]

        return MultiMetricResult(metrics=all_metrics)

    def cache_assets():
        _get_classifier()

    metric_names = [f"{input_name}.topics.{_sanitize_metric_name(topic)}" for topic in topics]
    return MultiMetric(names=metric_names, input_names=[input_name], evaluate=udf, cache_assets=cache_assets)


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
