from typing import Dict, List, Union
from langkit.translator import Translator, translated, translated_udf
import whylogs as why
from whylogs.core.metrics import FrequentItemsMetric
from whylogs.core.resolvers import MetricSpec
from whylogs.core.stubs import pd as pd
from whylogs.experimental.core.udf_schema import register_dataset_udf, udf_schema


class FakeTranslator(Translator):
    def translate(self, text: str) -> str:
        return text[::-1]


class NonTranslator(Translator):
    def translate(self, text: str) -> str:
        return text


@translated(FakeTranslator())
def translated_prompt(x):
    return x


def wrapper(input):
    return [translated_prompt(x) for x in input["prompt"]]


def test_translated_decorator():
    register_dataset_udf(
        ["prompt"],
        udf_name="prompt.translated",
        metrics=[MetricSpec(FrequentItemsMetric)],
        schema_name="translated",
    )(wrapper)
    untranslated = {"prompt": "This is a prompt", "response": "This is a response"}
    schema = udf_schema(schema_name="translated", include_default_schema=False)
    result = why.log(untranslated, schema=schema)
    assert (
        result.view()
        .get_column("prompt.translated")
        .get_metric("frequent_items")
        .strings[0]
        .value
        == untranslated["prompt"][::-1]
    )


@translated_udf({"prompt": NonTranslator(), "response": FakeTranslator()})
def concat(input: Union[Dict[str, List], pd.DataFrame]) -> Union[List, pd.Series]:
    return [x + y for x, y in zip(input["prompt"], input["response"])]


def test_translated_udf():
    register_dataset_udf(
        ["prompt", "response"],
        udf_name="bob",
        metrics=[MetricSpec(FrequentItemsMetric)],
        schema_name="translated",
    )(concat)
    untranslated = {"prompt": "knock knock", "response": "who's there?"}
    schema = udf_schema(schema_name="translated", include_default_schema=False)
    result = why.log(untranslated, schema=schema)
    assert (
        result.view().get_column("bob").get_metric("frequent_items").strings[0].value
        == untranslated["prompt"] + untranslated["response"][::-1]
    )
