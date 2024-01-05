from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Union

import pandas as pd
import torch
from transformers import Pipeline, pipeline  # type: ignore

from langkit.module.module import Module, UdfInput, UdfSchemaArgs
from langkit.module.util import LazyInit
from whylogs.core.resolvers import MetricSpec, ResolverSpec, StandardMetric
from whylogs.experimental.core.udf_schema import UdfSpec

__default_topics = ["politics", "economy", "entertainment", "environment"]

__classifier: LazyInit[Pipeline] = LazyInit(
    lambda: pipeline(
        "zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device="cuda" if torch.cuda.is_available() else "cpu"
    )
)


def __get_closest_topic(text: str, topics: List[str], multi_label: bool = False) -> str:
    return __classifier.value(text, topics, multi_label=multi_label)["labels"][0]  # type: ignore


def __topic_module(column_name: str, topics: List[str]) -> UdfSchemaArgs:
    def udf(text: Union[pd.DataFrame, Dict[str, List[Any]]]) -> Any:
        return [__get_closest_topic(it, topics) for it in UdfInput(text).iter_column_rows(column_name)]

    udf_column_name = f"{column_name}.closest_topic"

    spec = UdfSpec(
        column_names=[column_name],
        udfs={udf_column_name: udf},
    )

    return UdfSchemaArgs(
        types={column_name: str},
        resolvers=[
            ResolverSpec(
                column_name=udf_column_name,
                metrics=[MetricSpec(StandardMetric.frequent_items.value)],
            )
        ],
        udf_specs=[spec],
    )


prompt_topic_module = partial(__topic_module, "prompt", __default_topics)
response_topic_module = partial(__topic_module, "response", __default_topics)
prompt_response_topic_module = [prompt_topic_module, response_topic_module]


@dataclass
class CustomTopicModules:
    prompt_topic_module: Module
    response_topic_module: Module
    prompt_response_topic_module: Module


def get_custom_topic_modules(topics: List[str]) -> CustomTopicModules:
    prompt_topic_module = partial(__topic_module, "prompt", topics)
    response_topic_module = partial(__topic_module, "response", topics)
    return CustomTopicModules(
        prompt_topic_module=prompt_topic_module,
        response_topic_module=response_topic_module,
        prompt_response_topic_module=[prompt_topic_module, response_topic_module],
    )
