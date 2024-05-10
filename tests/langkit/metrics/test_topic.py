# pyright: reportUnknownMemberType=none
from typing import Any

import pandas as pd
import pytest

import whylogs as why
from langkit.core.metric import WorkflowMetricConfig, WorkflowMetricConfigBuilder
from langkit.core.workflow import Workflow
from langkit.metrics.library import lib
from langkit.metrics.topic import prompt_topic_module
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


def test_topic_wf():
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

    wf = Workflow(metrics=[lib.prompt.topics(topics=["economy", "entertainment", "medicine", "technology"], onnx=False)])
    wf_onnx = Workflow(metrics=[lib.prompt.topics(topics=["economy", "entertainment", "medicine", "technology"], onnx=True)])

    result = wf.run(df)
    result_onnx = wf_onnx.run(df)

    expected_columns = [
        "prompt.topics.economy",
        "prompt.topics.entertainment",
        "prompt.topics.medicine",
        "prompt.topics.technology",
        "id",
    ]

    print("result")
    print(result.metrics.transpose())

    print("result_onnx")
    print(result_onnx.metrics.transpose())

    assert result.metrics.columns.tolist() == expected_columns
    assert result_onnx.metrics.columns.tolist() == expected_columns
    assert_frames_approx_equal(result.metrics, result_onnx.metrics, tol=1e-5)


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

    # schema = WorkflowMetricConfigBuilder().add(prompt_topic_module).build()
    wf = Workflow(metrics=[prompt_topic_module])

    results = wf.run(df)
    print(results.metrics.transpose())

    expected_columns = [
        "prompt.topics.medicine",
        "prompt.topics.economy",
        "prompt.topics.technology",
        "prompt.topics.entertainment",
        "id",
    ]

    assert results.metrics.columns.tolist() == expected_columns


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
        assert actual.metrics[metric_name][0] < 0.5


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


def test_topic_library():
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

    topics = ["fishing", "boxing", "hiking", "swimming"]
    wf = Workflow(metrics=[lib.prompt.topics(topics), lib.response.topics(topics)])
    result = wf.run(df)
    actual = result.metrics

    expected_columns = [
        "prompt.topics.fishing",
        "prompt.topics.boxing",
        "prompt.topics.hiking",
        "prompt.topics.swimming",
        "response.topics.fishing",
        "response.topics.boxing",
        "response.topics.hiking",
        "response.topics.swimming",
        "id",
    ]
    assert actual.columns.tolist() == expected_columns

    pd.set_option("display.max_columns", None)
    print(actual.transpose())

    assert actual["prompt.topics.fishing"][0] > 0.50
    assert actual["prompt.topics.boxing"][1] > 0.50
    assert actual["prompt.topics.hiking"][2] > 0.50
    assert actual["prompt.topics.swimming"][3] > 0.50

    assert actual["response.topics.fishing"][0] > 0.50
    assert actual["response.topics.boxing"][1] > 0.50
    assert actual["response.topics.hiking"][2] > 0.50
    assert actual["response.topics.swimming"][3] > 0.50


def test_topic_name_sanitize():
    df = pd.DataFrame(
        {
            "prompt": [
                "What's the best kind of bait?",
            ],
            "response": [
                "The best kind of bait is worms.",
            ],
        }
    )

    topics = ["Fishing supplies"]
    wf = Workflow(metrics=[lib.prompt.topics(topics), lib.response.topics(topics)])

    result = wf.run(df)
    actual = result.metrics

    expected_columns = [
        "prompt.topics.fishing_supplies",
        "response.topics.fishing_supplies",
        "id",
    ]
    assert actual.columns.tolist() == expected_columns

    pd.set_option("display.max_columns", None)
    print(actual.transpose())

    assert actual["prompt.topics.fishing_supplies"][0] > 0.50


def assert_frames_approx_equal(df1: pd.DataFrame, df2: pd.DataFrame, tol=1e-6):
    assert df1.shape == df2.shape, "DataFrames are of different shapes"
    for col in df1.columns:  # type: ignore
        assert col in df2.columns, f"Column {col} missing from second DataFrame"
        for idx in df1.index:  # type: ignore
            assert idx in df2.index, f"Index {idx} missing from second DataFrame"
            val1 = df1.at[idx, col]  # type: ignore
            val2 = df2.at[idx, col]  # type: ignore
            assert val1 == pytest.approx(val2, abs=tol), f"Values at {idx}, {col} differ: {val1} != {val2}"  # type: ignore
