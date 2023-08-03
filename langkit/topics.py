from whylogs.experimental.core.udf_schema import register_type_udf
from whylogs.core.datatypes import String, Optional
from logging import getLogger
from transformers import (
    pipeline,
)
from . import LangKitConfig

diagnostic_logger = getLogger(__name__)

lang_config = LangKitConfig()
_topics = lang_config.topics

model_path = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
classifier = pipeline("zero-shot-classification", model=model_path)


@register_type_udf(String, namespace="privacy")
def closest_topic(strings) -> list:
    series_results = []
    for text in strings:
        try:
            output = classifier(text, _topics, multi_label=False)
            series_results.append(output['labels'][0])
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in closest topics udf: {e}"
            )
            series_results.append(None)
    return series_results


def init(topics: Optional[list] = None):
    global _topics
    if topics:
        _topics = topics


init()
