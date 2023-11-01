from copy import deepcopy
from whylogs.experimental.core.udf_schema import register_dataset_udf
from typing import Callable, List, Optional, Set
from transformers import (
    pipeline,
)
from . import LangKitConfig, lang_config, prompt_column, response_column
from langkit.whylogs.unreg import unregister_udfs


_topics: List[str] = lang_config.topics
_model_path: Optional[str] = None
_classifier = None


_response_topics: List[str] = lang_config.response_topics
_response_model_path: Optional[str] = None
_response_classifier = None


def closest_topic(text, classifier, topics):
    if classifier is None:
        raise ValueError("Topics - classifier model not initialized")
    return classifier(text, topics, multi_label=False)["labels"][0]


def _wrapper(column: str, classifier, topics) -> Callable:
    return lambda text: [closest_topic(t, classifier, topics) for t in text[column]]


_registered: Set[str] = set()


def init(
    language: Optional[str] = None,
    topics: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    topic_classifier: Optional[str] = None,
    config: Optional[LangKitConfig] = None,
    response_topics: Optional[List[str]] = None,
    response_model_path: Optional[str] = None,
    response_topic_classifier: Optional[str] = None,
):
    global _registered
    unregister_udfs(_registered)
    config = config or deepcopy(lang_config)
    global _topics, _classifier
    _topics = topics or config.topics
    topic_classifier = topic_classifier or lang_config.topic_classifier
    model_path = model_path or config.topic_model_path
    if not (model_path and topic_classifier):
        _classifier = None
    else:
        _classifier = pipeline(topic_classifier, model=model_path)

    global _response_topics, _response_classifier
    _response_topics = response_topics or config.response_topics
    topic_classifier = (
        response_topic_classifier or lang_config.response_topic_classifier
    )
    model_path = response_model_path or config.response_topic_model_path
    if not (model_path and topic_classifier):
        _response_classifier = None
    else:
        _response_classifier = pipeline(topic_classifier, model=model_path)

    if _classifier is not None:
        register_dataset_udf(
            [prompt_column], udf_name=f"{prompt_column}.closest_topic"
        )(_wrapper(prompt_column, _classifier, _topics))
        _registered.add(f"{prompt_column}.closest_topic")

    if _response_classifier is not None:
        register_dataset_udf(
            [response_column], udf_name=f"{response_column}.closest_topic"
        )(_wrapper(response_column, _response_classifier, _response_topics))
        _registered.add(f"{response_column}.closest_topic")
