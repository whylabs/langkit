# pyright: reportUnknownMemberType=none
from typing import Any

import pandas as pd

import whylogs as why
from langkit.core.metric import WorkflowMetricConfig, WorkflowMetricConfigBuilder
from langkit.core.workflow import Workflow
from langkit.metrics.topic import get_custom_topic_modules, prompt_topic_module
from langkit.metrics.whylogs_compat import create_whylogs_udf_schema

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


def _log(item: Any, conf: WorkflowMetricConfig) -> pd.DataFrame:
    schema = create_whylogs_udf_schema(conf)
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

    schema = WorkflowMetricConfigBuilder().add(prompt_topic_module).build()

    actual = _log(df, schema)

    expected_columns = [
        "prompt",
        "prompt.topics.economy",
        "prompt.topics.entertainment",
        "prompt.topics.medicine",
        "prompt.topics.technology",
        "response",
    ]

    assert actual.index.tolist() == expected_columns


def test_topic_empty_input():
    df = pd.DataFrame(
        {
            "prompt": [
                "",
            ],
            "response": [
                "George Washington is the president of the United States.",
            ],
        }
    )

    schema = WorkflowMetricConfigBuilder().add(prompt_topic_module).build()

    actual = _log(df, schema)

    expected_columns = [
        "prompt",
        "prompt.topics.economy",
        "prompt.topics.entertainment",
        "prompt.topics.medicine",
        "prompt.topics.technology",
        "response",
    ]

    assert actual.index.tolist() == expected_columns
    for column in expected_columns:
        if column not in ["prompt", "response"]:
            assert actual.loc[column]["counts/null"] == 1


def test_topic_empty_input_wf():
    df = pd.DataFrame(
        {
            "prompt": [
                "",
            ],
            "response": [
                "George Washington is the president of the United States.",
            ],
        }
    )
    expected_metrics = ["prompt.topics.economy", "prompt.topics.entertainment", "prompt.topics.medicine", "prompt.topics.technology"]
    wf = Workflow(metrics=[prompt_topic_module])
    actual = wf.run(df)
    for metric_name in expected_metrics:
        assert actual.metrics[metric_name][0] is None


def test_topic_row():
    row = {
        "prompt": "Who is the president of the United States?",
        "response": "George Washington is the president of the United States.",
    }

    schema = WorkflowMetricConfigBuilder().add(prompt_topic_module).build()

    actual = _log(row, schema)

    expected_columns = [
        "prompt",
        "prompt.topics.economy",
        "prompt.topics.entertainment",
        "prompt.topics.medicine",
        "prompt.topics.technology",
        "response",
    ]

    assert actual.index.tolist() == expected_columns


def test_custom_topic():
    df = pd.DataFrame(
        {
            "prompt": [
                "What's the best kind of bait?",
                "What's the best kind of punch?",
                "What's the best kind of trail?",
                "What's the best kind of swimming stroke?",
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
    schema = WorkflowMetricConfigBuilder().add(custom_topic_modules.prompt_response_topic_module).build()

    actual = _log(df, schema)

    expected_columns = [
        "prompt",
        "prompt.topics.boxing",
        "prompt.topics.fishing",
        "prompt.topics.hiking",
        "prompt.topics.swimming",
        "response",
        "response.topics.boxing",
        "response.topics.fishing",
        "response.topics.hiking",
        "response.topics.swimming",
    ]
    assert actual.index.tolist() == expected_columns
    for column in expected_columns:
        if column not in ["prompt", "response"]:
            assert actual.loc[column]["distribution/max"] >= 0.50
