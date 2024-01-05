from typing import Any

import pandas as pd

import whylogs as why
from langkit.module.module import SchemaBuilder
from langkit.module.topic import get_custom_topic_modules, prompt_topic_module
from whylogs.core.metrics.metrics import FrequentItem
from whylogs.core.schema import DatasetSchema

expected_metrics = [
    "cardinality/est",
    "cardinality/lower_1",
    "cardinality/upper_1",
    "counts/inf",
    "counts/n",
    "counts/nan",
    "counts/null",
    "distribution/max",
    "distribution/mean",
    "distribution/median",
    "distribution/min",
    "distribution/n",
    "distribution/q_01",
    "distribution/q_05",
    "distribution/q_10",
    "distribution/q_25",
    "distribution/q_75",
    "distribution/q_90",
    "distribution/q_95",
    "distribution/q_99",
    "distribution/stddev",
    "type",
    "types/boolean",
    "types/fractional",
    "types/integral",
    "types/object",
    "types/string",
    "types/tensor",
]


def _log(item: Any, schema: DatasetSchema) -> pd.DataFrame:
    return why.log(item, schema=schema).view().to_pandas()  # type: ignore


def test_topic():
    df = pd.DataFrame(
        {
            "prompt": [
                "Who is the president of the United States?",
                "What improves GDP?",
                "What causes global warming?",
                "Who was the star of the movie Titanic?",
            ],
            "response": [
                "George Washington is the president of the United States.",
                "GDP is improved by increasing the money supply.",
                "Global warming is caused by greenhouse gases.",
                "Leonardo DiCaprio was the star of the movie Titanic.",
            ],
        }
    )

    schema = SchemaBuilder().add(prompt_topic_module).build()

    actual = _log(df, schema)

    expected_columns = [
        "prompt",
        "prompt.closest_topic",
        "response",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["frequent_items/frequent_strings"]["prompt.closest_topic"] == [
        FrequentItem(value="environment", est=1, upper=1, lower=1),
        FrequentItem(value="politics", est=1, upper=1, lower=1),
        FrequentItem(value="entertainment", est=1, upper=1, lower=1),
        FrequentItem(value="economy", est=1, upper=1, lower=1),
    ]


def test_topic_row():
    row = {
        "prompt": "Who is the president of the United States?",
        "response": "George Washington is the president of the United States.",
    }

    schema = SchemaBuilder().add(prompt_topic_module).build()

    actual = _log(row, schema)

    expected_columns = [
        "prompt",
        "prompt.closest_topic",
        "response",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["frequent_items/frequent_strings"]["prompt.closest_topic"] == [
        FrequentItem(value="politics", est=1, upper=1, lower=1),
    ]


def test_custom_topic():
    df = pd.DataFrame(
        {
            "prompt": [
                "What's the best kind of bait?",
                "What's the best kind of punch?",
                "What's the best kind of trail?",
                "What's the best kind of stroke?",
            ],
            "response": [
                "The best kind of bait is worms.",
                "The best kind of punch is a jab.",
                "The best kind of trail is a loop.",
                "The best kind of stroke is freestyle.",
            ],
        }
    )

    custom_topic_modules = get_custom_topic_modules(["fishing", "boxing", "hiking", "swimming"])
    schema = SchemaBuilder().add(custom_topic_modules.prompt_response_topic_module).build()

    actual = _log(df, schema)

    expected_columns = [
        "prompt",
        "prompt.closest_topic",
        "response",
        "response.closest_topic",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["frequent_items/frequent_strings"]["prompt.closest_topic"] == [
        FrequentItem(value="boxing", est=1, upper=1, lower=1),
        FrequentItem(value="fishing", est=1, upper=1, lower=1),
        FrequentItem(value="swimming", est=1, upper=1, lower=1),
        FrequentItem(value="hiking", est=1, upper=1, lower=1),
    ]
    assert actual["frequent_items/frequent_strings"]["response.closest_topic"] == [
        FrequentItem(value="boxing", est=1, upper=1, lower=1),
        FrequentItem(value="fishing", est=1, upper=1, lower=1),
        FrequentItem(value="swimming", est=1, upper=1, lower=1),
        FrequentItem(value="hiking", est=1, upper=1, lower=1),
    ]
