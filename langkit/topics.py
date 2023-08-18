from whylogs.experimental.core.udf_schema import register_dataset_udf
from whylogs.core.stubs import pd
from typing import List, Optional
from transformers import (
    pipeline,
)
from . import lang_config, prompt_column, response_column


_topics = lang_config.topics

model_path = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
classifier = pipeline("zero-shot-classification", model=model_path)


def closest_topic(text):
    index = text.columns[0] if isinstance(text, pd.DataFrame) else list(text.keys())[0]
    result = []
    for input in text[index]:
        output = classifier(input, _topics, multi_label=False)
        result.append(output["labels"][0])
    return result


def init(topics: Optional[List] = None):
    global _topics
    if topics:
        _topics = topics
    for column in [prompt_column, response_column]:
        register_dataset_udf([column], udf_name=f"{column}.closest_topic")(
            closest_topic
        )


init()
