from typing import Any

import pandas as pd

import whylogs as why
from langkit.module.input_output_similarity import input_output_similarity_module
from langkit.module.metric import EvaluationConfifBuilder, EvaluationConfig
from langkit.module.whylogs_compat import create_whylogs_udf_schema

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


def _log(item: Any, conf: EvaluationConfig) -> pd.DataFrame:
    schema = create_whylogs_udf_schema(conf)
    return why.log(item, schema=schema).view().to_pandas()  # type: ignore


def test_input_output():
    df = pd.DataFrame(
        {
            "prompt": [
                "I'm going to ask you a question",
            ],
            "response": [
                "I'm going to answer that question!",
            ],
        }
    )

    schema = EvaluationConfifBuilder().add(input_output_similarity_module).build()

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "response",
        "response.relevance_to_prompt",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["response.relevance_to_prompt"] > 0.5
    assert actual["distribution/min"]["response.relevance_to_prompt"] > 0.5
    assert actual["types/fractional"]["response.relevance_to_prompt"] == 1
    assert actual["cardinality/est"]["response.relevance_to_prompt"] == 1


def test_input_output_row():
    row = {
        "prompt": "I'm going to ask you a question",
        "response": "I'm going to answer that question!",
    }

    schema = EvaluationConfifBuilder().add(input_output_similarity_module).build()

    actual = _log(row, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "response",
        "response.relevance_to_prompt",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["response.relevance_to_prompt"] > 0.5
    assert actual["distribution/min"]["response.relevance_to_prompt"] > 0.5
    assert actual["types/fractional"]["response.relevance_to_prompt"] == 1
    assert actual["cardinality/est"]["response.relevance_to_prompt"] == 1


def test_input_output_multiple():
    df = pd.DataFrame(
        {
            "prompt": [
                "If you understand me, say 'yes'",
                "Ask me a question",
            ],
            "response": [
                "yes",
                "the sky is eating",
            ],
        }
    )

    schema = EvaluationConfifBuilder().add(input_output_similarity_module).build()

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "response",
        "response.relevance_to_prompt",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["response.relevance_to_prompt"] > 0
    assert actual["distribution/min"]["response.relevance_to_prompt"] > 0
    assert actual["types/fractional"]["response.relevance_to_prompt"] == 2
    assert int(actual["cardinality/est"]["response.relevance_to_prompt"]) == 2  # type: ignore
