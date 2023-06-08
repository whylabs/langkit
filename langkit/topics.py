from whylogs.experimental.core.metrics.udf_metric import register_metric_udf
from whylogs.core.datatypes import String, Optional
from transformers import (
    pipeline,
)
from . import LangKitConfig

lang_config = LangKitConfig()
_topics = lang_config.topics

model_path = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
classifier = pipeline("zero-shot-classification", model=model_path)


@register_metric_udf(col_type=String)
def closest_topic(text: str) -> str:
    output = classifier(text, _topics, multi_label=False)
    return output["labels"][0]


def init(topics: Optional[list] = None):
    global _topics
    if topics:
        _topics = topics


init()
