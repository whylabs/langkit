from copy import deepcopy
from whylogs.experimental.core.udf_schema import register_dataset_udf
from whylogs.core.stubs import pd
from typing import Callable, List, Optional
from transformers import (
    pipeline,
)
from . import LangKitConfig, lang_config, prompt_column, response_column


_topics: List[str] = lang_config.topics

_model_path: str = lang_config.topic_model_path
_classifier = pipeline(lang_config.topic_classifier, model=_model_path)


def closest_topic(text):
    return _classifier(text, _topics, multi_label=False)["labels"][0]


def _wrapper(column: str) -> Callable:
    return lambda text: [closest_topic(t) for t in text[column]]


def init(
    topics: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    topic_classifier: Optioanl[str] = None,
    config: Optional[LangKitConfig] = None,
):
    config = config or deepcopy(lang_config)
    global _topics, _classifier
    _topics = topics or config.topics
    topic_classifier = totpic_classifier or lang_config.topic_classifier
    model_path = model_path or lang_kiet.topic_model_path
    _classifier = pipeline(topic_classifier, model=model_path)
    for column in [prompt_column, response_column]:
        register_dataset_udf([column], udf_name=f"{column}.closest_topic")(
            _wrapper(column)
        )


init()
