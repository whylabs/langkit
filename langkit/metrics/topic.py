# pyright: reportUnknownMemberType=none
# pyright: reportUnknownVariableType=none
# pyright: reportUnknownLambdaType=none

import os
from functools import lru_cache, partial
from typing import List, Optional, TypedDict

import pandas as pd
import torch
from optimum.modeling_base import PreTrainedModel
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, Pipeline, PreTrainedTokenizerBase, pipeline  # type: ignore

from langkit.core.metric import MultiMetric, MultiMetricResult, UdfInput


class ClassificationResults(TypedDict):
    sequence: str
    labels: List[str]
    scores: List[float]


__default_topics = [
    "medicine",
    "economy",
    "technology",
    "entertainment",
]

_hypothesis_template = "This example is about {}"


def _get_classifier_normal(model: str, revision: str):
    return pipeline(
        "zero-shot-classification",
        model=model,
        revision=revision,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def _download_assets_onnx(model: str, revision: str):
    ORTModelForSequenceClassification.from_pretrained(
        model,
        subfolder="onnx",
        file_name="model.onnx",
        revision=revision,
        export=False,
    )
    AutoTokenizer.from_pretrained(model, revision=revision)


def _get_tokenizer(model: str, revision: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)


def _get_model(model: str, revision: str) -> PreTrainedModel:
    # return ORTModelForSequenceClassification.from_pretrained(
    #     _model,
    #     subfolder="onnx",
    #     file_name="model.onnx",
    #     export=False,
    #     revision=_revision,
    #     local_files_only=True,
    # )
    # Optimum doesn't support offline mode https://github.com/huggingface/optimum/issues/1796
    # workaround for now is to reference the actual model path after caching it. Uncomment the above code when the issue is resolved

    model_name = model.replace("/", "--")
    home_dir = os.path.expanduser("~")
    base = os.environ.get("HF_HOME", os.path.join(home_dir, ".cache/huggingface"))
    model_path = f"{base}/hub/models--{model_name }/snapshots/{revision}"
    return ORTModelForSequenceClassification.from_pretrained(
        model_path,
        file_name="onnx/model.onnx",
        export=False,
        revision=revision,
        local_files_only=True,
    )


@lru_cache
def _get_classifier_onnx(model: str, revision: str) -> Pipeline:
    return pipeline(
        "zero-shot-classification",
        model=_get_model(model, revision),  # pyright: ignore[reportArgumentType]
        tokenizer=_get_tokenizer(model, revision),  # pyright: ignore[reportArgumentType]
        truncation=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def __sanitize_text(text: str) -> str:
    """
    Sanitize input to the classifier pipeline. This model chokes on empty strings so we just map those
    to a single space to avoid having to generate mock scores for select items or bailing out of the
    batch based api (list of strings) and losing potential performance gains.
    """
    if text == "":
        return " "

    return text


def __get_scores_per_label(
    classifier: Pipeline, text: List[str], topics: List[str], hypothesis_template: str = _hypothesis_template, multi_label: bool = True
) -> List[ClassificationResults]:
    if not text:
        return []

    sanitized_text = [__sanitize_text(it) for it in text]

    result: List[ClassificationResults] = classifier(
        sanitized_text, topics, hypothesis_template=hypothesis_template, multi_label=multi_label
    )  # type: ignore
    return result


def _sanitize_metric_name(topic: str) -> str:
    """
    sanitize a metric name created from a topic. Replace white space with underscores, etc.
    """
    return topic.replace(" ", "_").lower()


def topic_metric(
    input_name: str,
    topics: List[str],
    hypothesis_template: Optional[str] = None,
    model: Optional[str] = None,
    revision: Optional[str] = None,
    use_onnx: bool = True,
) -> MultiMetric:
    hypothesis_template = hypothesis_template or _hypothesis_template
    model = "MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33" if model is None else model
    revision = "dea69e79cd6063916d08b883ea8a3c1823fd10b4" if revision is None else revision

    def udf(text: pd.DataFrame) -> MultiMetricResult:
        value: List[str] = list(UdfInput(text).iter_column_rows(input_name))

        if use_onnx:
            classifier = _get_classifier_onnx(model, revision)
        else:
            classifier = _get_classifier_normal(model, revision)

        results = __get_scores_per_label(classifier, value, topics=topics, hypothesis_template=hypothesis_template)

        all_metrics: List[List[float]] = [[] for _ in topics]
        for result in results:
            # Map each topic to its score in the current result
            topic_to_score = {label: score for label, score in zip(result["labels"], result["scores"])}
            # For each topic, append the score to the corresponding list in all_metrics
            for i, topic in enumerate(topics):
                all_metrics[i].append(topic_to_score[topic])  # Append list of score for the topic

        return MultiMetricResult(metrics=all_metrics)

    def cache_assets():
        if use_onnx:
            _download_assets_onnx(model, revision)
        else:
            _get_classifier_normal(model, revision)

    def init():
        if use_onnx:
            _get_classifier_onnx(model, revision)
        else:
            _get_classifier_normal(model, revision)

    metric_names = [f"{input_name}.topics.{_sanitize_metric_name(topic)}" for topic in topics]
    metadata = {"model": model, "revision": revision}
    return MultiMetric(names=metric_names, input_names=[input_name], evaluate=udf, cache_assets=cache_assets, init=init, metadata=metadata)


prompt_topic_module = partial(topic_metric, "prompt", __default_topics, _hypothesis_template)
response_topic_module = partial(topic_metric, "response", __default_topics, _hypothesis_template)
prompt_response_topic_module = [prompt_topic_module, response_topic_module, _hypothesis_template]
