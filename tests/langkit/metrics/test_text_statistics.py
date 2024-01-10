# pyright: reportUnknownMemberType=none
from typing import Any

import pandas as pd

import whylogs as why
from langkit.metrics.metric import EvaluationConfifBuilder, EvaluationConfig
from langkit.metrics.text_statistics import (
    prompt_char_count_module,
    prompt_reading_ease_module,
    prompt_response_flesch_kincaid_grade_level_module,
    prompt_response_textstat_module,
    prompt_textstat_module,
    response_char_count_module,
    response_reading_ease_module,
    response_textstat_module,
)
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
    "ints/max",
    "ints/min",
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


def _log(item: Any, conf: EvaluationConfig) -> pd.DataFrame:
    schema = create_whylogs_udf_schema(conf)
    return why.log(item, schema=schema).view().to_pandas()  # type: ignore


def test_prompt_response_textstat_module():
    all_textstat_schema = EvaluationConfifBuilder().add(prompt_response_textstat_module).build()

    actual = _log(row, all_textstat_schema)

    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.char_count",
        "prompt.difficult_words",
        "prompt.flesch_kincaid_grade",
        "prompt.flesch_reading_ease",
        "prompt.letter_count",
        "prompt.lexicon_count",
        "prompt.monosyllabcount",
        "prompt.polysyllabcount",
        "prompt.sentence_count",
        "prompt.syllable_count",
        "response",
        "response.char_count",
        "response.difficult_words",
        "response.flesch_kincaid_grade",
        "response.flesch_reading_ease",
        "response.letter_count",
        "response.lexicon_count",
        "response.monosyllabcount",
        "response.polysyllabcount",
        "response.sentence_count",
        "response.syllable_count",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.char_count"] == len(row["prompt"].replace(" ", ""))
    assert actual["distribution/max"]["response.char_count"] == len(row["response"].replace(" ", ""))

    actual_row = _log(row, all_textstat_schema)

    assert actual_row.index.tolist() == expected_columns
    assert actual_row["distribution/max"]["prompt.char_count"] == len(row["prompt"].replace(" ", ""))
    assert actual_row["distribution/max"]["response.char_count"] == len(row["response"].replace(" ", ""))


def test_prompt_textstat_module():
    prompt_textstat_schema = EvaluationConfifBuilder().add(prompt_textstat_module).build()

    actual = _log(row, prompt_textstat_schema)

    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.char_count",
        "prompt.difficult_words",
        "prompt.flesch_kincaid_grade",
        "prompt.flesch_reading_ease",
        "prompt.letter_count",
        "prompt.lexicon_count",
        "prompt.monosyllabcount",
        "prompt.polysyllabcount",
        "prompt.sentence_count",
        "prompt.syllable_count",
        "response",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.char_count"] == len(row["prompt"].replace(" ", ""))
    assert "response.char_count" not in actual["distribution/max"]

    actual_row = _log(row, prompt_textstat_schema)

    assert actual_row.index.tolist() == expected_columns
    assert actual_row["distribution/max"]["prompt.char_count"] == len(row["prompt"].replace(" ", ""))
    assert "response.char_count" not in actual_row["distribution/max"]


def test_response_textstat_module():
    response_textstat_schema = EvaluationConfifBuilder().add(response_textstat_module).build()

    actual = _log(row, response_textstat_schema)

    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "response",
        "response.char_count",
        "response.difficult_words",
        "response.flesch_kincaid_grade",
        "response.flesch_reading_ease",
        "response.letter_count",
        "response.lexicon_count",
        "response.monosyllabcount",
        "response.polysyllabcount",
        "response.sentence_count",
        "response.syllable_count",
    ]

    assert actual.index.tolist() == expected_columns
    assert "prompt.char_count" not in actual["distribution/max"]
    assert actual["distribution/max"]["response.char_count"] == len(row["response"].replace(" ", ""))

    actual_row = _log(row, response_textstat_schema)

    assert actual_row.index.tolist() == expected_columns
    assert "prompt.char_count" not in actual_row["distribution/max"]
    assert actual_row["distribution/max"]["response.char_count"] == len(row["response"].replace(" ", ""))


def test_prompt_reading_ease_module():
    prompt_reading_ease_schema = EvaluationConfifBuilder().add(prompt_reading_ease_module).build()

    actual = _log(row, prompt_reading_ease_schema)

    # score is a float so it doesn't have the ints metrics
    expected_metrics_without_ints = [metric for metric in expected_metrics if "ints" not in metric]

    assert list(actual.columns) == expected_metrics_without_ints

    assert actual.index.tolist() == [
        "prompt",
        "prompt.flesch_reading_ease",
        "response",
    ]


def test_response_reading_ease_module():
    response_reading_ease_schema = EvaluationConfifBuilder().add(response_reading_ease_module).build()

    actual = _log(row, response_reading_ease_schema)

    # score is a float so it doesn't have the ints metrics
    expected_metrics_without_ints = [metric for metric in expected_metrics if "ints" not in metric]

    assert list(actual.columns) == expected_metrics_without_ints

    assert actual.index.tolist() == [
        "prompt",
        "response",
        "response.flesch_reading_ease",
    ]


def test_prompt_response_flesch_kincaid_grade_level_module():
    schema = EvaluationConfifBuilder().add(prompt_response_flesch_kincaid_grade_level_module).build()

    actual = _log(row, schema)

    # score is a float so it doesn't have the ints metrics
    expected_metrics_without_ints = [metric for metric in expected_metrics if "ints" not in metric]

    assert list(actual.columns) == expected_metrics_without_ints

    assert actual.index.tolist() == [
        "prompt",
        "prompt.flesch_kincaid_grade",
        "response",
        "response.flesch_kincaid_grade",
    ]


def test_prompt_char_count_module():
    prompt_char_count_schema = EvaluationConfifBuilder().add(prompt_char_count_module).build()

    actual = _log(row, prompt_char_count_schema)

    assert list(actual.columns) == expected_metrics

    assert actual.index.tolist() == [
        "prompt",
        "prompt.char_count",
        "response",
    ]


def test_response_char_count_module():
    response_char_count_schema = EvaluationConfifBuilder().add(response_char_count_module).build()

    actual = _log(row, response_char_count_schema)

    assert list(actual.columns) == expected_metrics

    assert actual.index.tolist() == [
        "prompt",
        "response",
        "response.char_count",
    ]


def test_custom_module_combination():
    from langkit.metrics.text_statistics import (
        prompt_char_count_module,
        prompt_difficult_words_module,
        prompt_reading_ease_module,
        response_char_count_module,
        response_sentence_count_module,
    )

    schema = (
        EvaluationConfifBuilder()
        .add(prompt_char_count_module)
        .add(prompt_reading_ease_module)
        .add(prompt_difficult_words_module)
        .add(response_char_count_module)
        .add(response_sentence_count_module)
        .build()
    )

    actual = _log(row, schema)

    expected_columns = [
        "prompt",
        "prompt.char_count",
        "prompt.difficult_words",
        "prompt.flesch_reading_ease",
        "response",
        "response.char_count",
        "response.sentence_count",
    ]

    assert list(actual.columns) == expected_metrics
    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.char_count"] == len(row["prompt"].replace(" ", ""))
    assert actual["distribution/max"]["response.char_count"] == len(row["response"].replace(" ", ""))

    # and you get the same results if you combine the modules in different ways

    prompt_modules = [
        prompt_char_count_module,
        prompt_reading_ease_module,
        prompt_difficult_words_module,
    ]

    response_modules = [
        response_char_count_module,
        response_sentence_count_module,
    ]

    schema = EvaluationConfifBuilder().add(prompt_modules).add(response_modules).build()

    actual = _log(row, schema)

    assert list(actual.columns) == expected_metrics
    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.char_count"] == len(row["prompt"].replace(" ", ""))
    assert actual["distribution/max"]["response.char_count"] == len(row["response"].replace(" ", ""))
