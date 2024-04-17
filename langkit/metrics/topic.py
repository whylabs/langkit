from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from transformers import Pipeline, pipeline  # type: ignore

from langkit.core.metric import MetricCreator, MultiMetric, MultiMetricResult

__default_topics = [
    "medicine",
    "economy",
    "technology",
    "entertainment",
]

_hypothesis_template = "This example is about {}"


@lru_cache(maxsize=None)
def _get_classifier(model_version: str) -> Pipeline:
    return pipeline(
        "zero-shot-classification",
        model=model_version,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


MODEL_SMALL = "MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33"
MODEL_BASE = "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33"
MODEL_LARGE = "MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33"


def __get_scores_per_label(
    text: str,
    topics: List[str],
    hypothesis_template: str = _hypothesis_template,
    multi_label: bool = True,
    model_version: str = MODEL_SMALL,
) -> Optional[Dict[str, float]]:
    if not text:
        return None
    result: Dict[str, [str, float]] = _get_classifier(model_version)(  # type: ignore
        text,
        topics,
        hypothesis_template=hypothesis_template,
        multi_label=multi_label,  # type: ignore
    )
    scores_per_label: Dict[str, float] = {label: score for label, score in zip(result["labels"], result["scores"])}  # type: ignore[reportUnknownVariableType]
    return scores_per_label


def _sanitize_metric_name(topic: str) -> str:
    """
    sanitize a metric name created from a topic. Replace white space with underscores, etc.
    """
    return topic.replace(" ", "_").lower()


def topic_metric(
    input_name: str, topics: List[str], hypothesis_template: Optional[str] = None, model_version: str = MODEL_SMALL
) -> MultiMetric:
    hypothesis_template = hypothesis_template or _hypothesis_template

    def udf(text: pd.DataFrame) -> MultiMetricResult:
        metrics: Dict[str, List[Optional[float]]] = {topic: [] for topic in topics}

        def process_row(row: pd.DataFrame) -> Dict[str, List[Optional[float]]]:
            value: Any = row[input_name]  # type: ignore
            scores = __get_scores_per_label(value, topics=topics, hypothesis_template=hypothesis_template, model_version=model_version)  # pyright: ignore[reportUnknownArgumentType]
            for topic in topics:
                metrics[topic].append(scores[topic] if scores else None)
            return metrics

        text.apply(process_row, axis=1)  # pyright: ignore[reportUnknownMemberType]

        all_metrics = [
            *metrics.values(),
        ]

        return MultiMetricResult(metrics=all_metrics)

    def cache_assets():
        _get_classifier(model_version)

    metric_names = [f"{input_name}.topics.{_sanitize_metric_name(topic)}" for topic in topics]
    return MultiMetric(names=metric_names, input_names=[input_name], evaluate=udf, cache_assets=cache_assets)


prompt_topic_module = partial(topic_metric, "prompt", __default_topics, _hypothesis_template, MODEL_SMALL)
response_topic_module = partial(topic_metric, "response", __default_topics, _hypothesis_template, MODEL_SMALL)
prompt_response_topic_module = [prompt_topic_module, response_topic_module, _hypothesis_template, MODEL_SMALL]


@dataclass
class CustomTopicModules:
    prompt_topic_module: MetricCreator
    response_topic_module: MetricCreator
    prompt_response_topic_module: MetricCreator


def get_custom_topic_modules(
    topics: List[str], template: str = _hypothesis_template, model_version: str = MODEL_SMALL
) -> CustomTopicModules:
    prompt_topic_module = partial(topic_metric, "prompt", topics, template, model_version)
    response_topic_module = partial(topic_metric, "response", topics, template, model_version)
    return CustomTopicModules(
        prompt_topic_module=prompt_topic_module,
        response_topic_module=response_topic_module,
        prompt_response_topic_module=[prompt_topic_module, response_topic_module],
    )
