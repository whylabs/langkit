# pyright: reportUnknownMemberType=none
from typing import Any

import pandas as pd

import whylogs as why
from langkit.core.metric import WorkflowMetricConfig, WorkflowMetricConfigBuilder
from langkit.core.workflow import Workflow
from langkit.metrics.library import lib as metrics_lib
from langkit.metrics.toxicity import prompt_response_toxicity_module, prompt_toxicity_metric, response_toxicity_metric
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


df = pd.DataFrame(
    {
        "prompt": [
            "Hi, how are you doing today?",
            "Hi, how are you doing today?",
            "Hi, how are you doing today?",
            "Hi, how are you doing today?",
        ],
        "response": [
            "I'm doing great, how about you?",
            "I'm doing great, how about you?",
            "I'm doing great, how about you?",
            "I'm doing great, how about you?",
        ],
    }
)

row = {"prompt": "Hi, how are you doing today?", "response": "I'm doing great, how about you?"}


def _log(item: Any, conf: WorkflowMetricConfig) -> pd.DataFrame:
    schema = create_whylogs_udf_schema(conf)
    return why.log(item, schema=schema).view().to_pandas()  # type: ignore


def test_prompt_toxicity_row_non_toxic():
    schema = WorkflowMetricConfigBuilder().add(prompt_toxicity_metric).build()

    actual = _log(row, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.toxicity.toxicity_score",
        "response",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.toxicity.toxicity_score"] < 0.1


def test_prompt_toxicity_version():
    wf = Workflow(metrics=[metrics_lib.prompt.toxicity.toxicity_score(hf_model_revision="f1c3aa41130e8baeee31c3ea5d14598a0d3385e5")])
    result = wf.run(row)

    expected_columns = ["prompt.toxicity.toxicity_score", "id"]

    assert list(result.metrics.columns) == expected_columns


def test_prompt_toxicity_df_non_toxic():
    schema = WorkflowMetricConfigBuilder().add(prompt_toxicity_metric).build()

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.toxicity.toxicity_score",
        "response",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.toxicity.toxicity_score"] < 0.1


def test_prompt_toxicity_row_toxic():
    schema = WorkflowMetricConfigBuilder().add(prompt_toxicity_metric).build()

    row = {"prompt": "You're a terrible person", "response": "I'm doing great, how about you?"}

    actual = _log(row, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.toxicity.toxicity_score",
        "response",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.toxicity.toxicity_score"] > 0.7


def test_prompt_toxicity_df_toxic():
    schema = WorkflowMetricConfigBuilder().add(prompt_toxicity_metric).build()

    df = pd.DataFrame(
        {
            "prompt": [
                "You're a terrible person",
                "You're a terrible person",
                "You're a terrible person",
                "You're a terrible person",
            ],
            "response": [
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
            ],
        }
    )

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.toxicity.toxicity_score",
        "response",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.toxicity.toxicity_score"] > 0.7


def test_prompt_toxicity_df_mixed():
    schema = WorkflowMetricConfigBuilder().add(prompt_toxicity_metric).build()

    df = pd.DataFrame(
        {
            "prompt": [
                "You're a terrible person",
                "Hi, how are you doing today?",
                "You're a terrible person",
                "Hi, how are you doing today?",
            ],
            "response": [
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
            ],
        }
    )

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.toxicity.toxicity_score",
        "response",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.toxicity.toxicity_score"] > 0.7
    assert actual["distribution/min"]["prompt.toxicity.toxicity_score"] < 0.1


def test_response_toxicity_row_non_toxic():
    schema = WorkflowMetricConfigBuilder().add(response_toxicity_metric).build()

    actual = _log(row, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "response",
        "response.toxicity.toxicity_score",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["response.toxicity.toxicity_score"] < 0.1


def test_response_toxicity_df_non_toxic():
    schema = WorkflowMetricConfigBuilder().add(response_toxicity_metric).build()

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "response",
        "response.toxicity.toxicity_score",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["response.toxicity.toxicity_score"] < 0.1


def test_response_toxicity_row_toxic():
    schema = WorkflowMetricConfigBuilder().add(response_toxicity_metric).build()

    row = {"prompt": "Hi, how are you doing today?", "response": "You're a terrible person"}

    actual = _log(row, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "response",
        "response.toxicity.toxicity_score",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["response.toxicity.toxicity_score"] > 0.7


def test_response_toxicity_df_toxic():
    schema = WorkflowMetricConfigBuilder().add(response_toxicity_metric).build()

    df = pd.DataFrame(
        {
            "prompt": [
                "Hi, how are you doing today?",
                "Hi, how are you doing today?",
                "Hi, how are you doing today?",
                "Hi, how are you doing today?",
            ],
            "response": [
                "You're a terrible person",
                "You're a terrible person",
                "You're a terrible person",
                "You're a terrible person",
            ],
        }
    )

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "response",
        "response.toxicity.toxicity_score",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["response.toxicity.toxicity_score"] > 0.7


def test_response_toxicity_df_mixed():
    schema = WorkflowMetricConfigBuilder().add(response_toxicity_metric).build()

    df = pd.DataFrame(
        {
            "prompt": [
                "Hi, how are you doing today?",
                "Hi, how are you doing today?",
                "Hi, how are you doing today?",
                "Hi, how are you doing today?",
            ],
            "response": [
                "I'm doing great, how about you?",
                "You're a terrible person",
                "I'm doing great, how about you?",
                "You're a terrible person",
            ],
        }
    )

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "response",
        "response.toxicity.toxicity_score",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["response.toxicity.toxicity_score"] > 0.7
    assert actual["distribution/min"]["response.toxicity.toxicity_score"] < 0.1


def test_prompt_response_toxicity_row_non_toxic():
    schema = WorkflowMetricConfigBuilder().add(prompt_response_toxicity_module).build()

    actual = _log(row, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.toxicity.toxicity_score",
        "response",
        "response.toxicity.toxicity_score",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.toxicity.toxicity_score"] < 0.1
    assert actual["distribution/max"]["response.toxicity.toxicity_score"] < 0.1


def test_prompt_response_toxicity_df_non_toxic():
    schema = WorkflowMetricConfigBuilder().add(prompt_response_toxicity_module).build()

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.toxicity.toxicity_score",
        "response",
        "response.toxicity.toxicity_score",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.toxicity.toxicity_score"] < 0.1
    assert actual["distribution/max"]["response.toxicity.toxicity_score"] < 0.1


def test_prompt_response_toxicity_row_toxic():
    schema = WorkflowMetricConfigBuilder().add(prompt_response_toxicity_module).build()

    row = {"prompt": "You're a terrible person", "response": "You're a terrible person"}

    actual = _log(row, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.toxicity.toxicity_score",
        "response",
        "response.toxicity.toxicity_score",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.toxicity.toxicity_score"] > 0.7
    assert actual["distribution/max"]["response.toxicity.toxicity_score"] > 0.7


def test_prompt_response_toxicity_df_toxic():
    schema = WorkflowMetricConfigBuilder().add(prompt_response_toxicity_module).build()

    df = pd.DataFrame(
        {
            "prompt": [
                "You're a terrible person",
                "You're a terrible person",
                "You're a terrible person",
                "You're a terrible person",
            ],
            "response": [
                "You're a terrible person",
                "You're a terrible person",
                "You're a terrible person",
                "You're a terrible person",
            ],
        }
    )

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.toxicity.toxicity_score",
        "response",
        "response.toxicity.toxicity_score",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.toxicity.toxicity_score"] > 0.7
    assert actual["distribution/min"]["prompt.toxicity.toxicity_score"] > 0.7
    assert actual["distribution/max"]["response.toxicity.toxicity_score"] > 0.7
    assert actual["distribution/min"]["response.toxicity.toxicity_score"] > 0.7


def test_prompt_response_toxicity_df_mixed():
    schema = WorkflowMetricConfigBuilder().add(prompt_response_toxicity_module).build()

    df = pd.DataFrame(
        {
            "prompt": [
                "You're a terrible person",
                "Hi, how are you doing today?",
                "You're a terrible person",
                "Hi, how are you doing today?",
            ],
            "response": [
                "You're a terrible person",
                "I'm doing great, how about you?",
                "You're a terrible person",
                "I'm doing great, how about you?",
            ],
        }
    )

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.toxicity.toxicity_score",
        "response",
        "response.toxicity.toxicity_score",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.toxicity.toxicity_score"] > 0.7
    assert actual["distribution/min"]["prompt.toxicity.toxicity_score"] < 0.1
    assert actual["distribution/max"]["response.toxicity.toxicity_score"] > 0.7
    assert actual["distribution/min"]["response.toxicity.toxicity_score"] < 0.1
