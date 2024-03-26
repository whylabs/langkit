# pyright: reportUnknownMemberType=none
import json
import math
import os
import re
import tempfile
from typing import Any

import pandas as pd

import whylogs as why
from langkit.core.metric import WorkflowMetricConfig, WorkflowMetricConfigBuilder
from langkit.core.workflow import Workflow
from langkit.metrics.regexes.regex_loader import CompiledPatternGroups, PatternGroups
from langkit.metrics.regexes.regexes import (
    get_custom_regex_frequent_items_for_column_module,
    get_custom_regex_frequent_items_modules,
    get_custom_regex_modules,
    prompt_credit_card_number_regex_metric,
    prompt_default_regexes_enum_metric,
    prompt_email_address_regex_metric,
    prompt_mailing_address_regex_metric,
    prompt_phone_number_regex_metric,
    prompt_response_credit_card_number_regex_module,
    prompt_response_default_regexes_module,
    prompt_response_email_address_regex_module,
    prompt_response_mailing_address_regex_module,
    prompt_response_phone_number_regex_module,
    prompt_response_ssn_regex_module,
    prompt_ssn_regex_metric,
    prompt_url_regex_metric,
    response_credit_card_number_regex_metric,
    response_default_regexes_module,
    response_email_address_regex_metric,
    response_mailing_address_regex_metric,
    response_phone_number_regex_metric,
    response_ssn_regex_metric,
    response_url_regex_metric,
)
from langkit.metrics.whylogs_compat import create_whylogs_udf_schema
from whylogs.core.metrics.metrics import FrequentItem

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


def _log(item: Any, conf: WorkflowMetricConfig) -> pd.DataFrame:
    schema = create_whylogs_udf_schema(conf)
    return why.log(item, schema=schema).view().to_pandas()  # type: ignore


def test_prompt_regex_df_url():
    df = pd.DataFrame(
        {
            "prompt": [
                "Does this code look good? foo.strip()/10. My blog is at whylabs.ai/foo.html",
            ],
            "response": [
                "Yeah. Nice blog, mine is at whylabs.ai/bar.html",
            ],
        }
    )

    wf = Workflow(metrics=[prompt_url_regex_metric, response_url_regex_metric])
    result = wf.run(df)

    actual = result.metrics

    expected_columns = ["prompt.regex.url", "response.regex.url", "id"]

    assert list(actual.columns) == expected_columns
    assert actual["prompt.regex.url"][0] == 1
    assert actual["response.regex.url"][0] == 1


def test_prompt_regex_df_ssn():
    df = pd.DataFrame(
        {
            "prompt": [
                "This is a ssn: 123-45-6789",
                "This is a ssn: 119-45-6789",
                "This is a ssn: 123-45-6789",
                "This is a ssn: redacted",
            ],
            "response": [
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
            ],
        }
    )

    schema = WorkflowMetricConfigBuilder().add(prompt_ssn_regex_metric).build()

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.regex.ssn",
        "response",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.regex.ssn"] == 1
    assert actual["distribution/min"]["prompt.regex.ssn"] == 0
    assert actual["types/integral"]["prompt.regex.ssn"] == 4


def test_response_regex_df_ssn():
    df = pd.DataFrame(
        {
            "prompt": [
                "How are you?",
                "How are you?",
                "How are you?",
                "How are you?",
            ],
            "response": [
                "I'm doing great, here's my ssn: 123-45-6789",
                "I'm doing great, here's my ssn: 123-45-6789",
                "I'm doing great, here's my ssn: 123-45-6789",
                "I'm doing great, how about you?",
            ],
        }
    )

    schema = WorkflowMetricConfigBuilder().add(response_ssn_regex_metric).build()

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "response",
        "response.regex.ssn",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["response.regex.ssn"] == 1
    assert actual["distribution/min"]["response.regex.ssn"] == 0
    assert actual["types/integral"]["response.regex.ssn"] == 4


def test_response_regex_df_ssn_row():
    row = {
        "prompt": "How are you?",
        "response": "I'm doing great, here's my ssn: 123-45-6789",
    }

    schema = WorkflowMetricConfigBuilder().add(response_ssn_regex_metric).build()

    actual = _log(row, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "response",
        "response.regex.ssn",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["response.regex.ssn"] == 1
    assert actual["distribution/min"]["response.regex.ssn"] == 1
    assert actual["types/integral"]["response.regex.ssn"] == 1


def test_prompt_response_df_ssn():
    df = pd.DataFrame(
        {
            "prompt": [
                "This is a ssn: 123-45-6789",
                "This is a ssn: 119-45-6789",
                "This is a ssn: 123-45-6789",
                "This is a ssn: redacted",
            ],
            "response": [
                "I'm doing great, here's my ssn: 123-45-6789",
                "I'm doing great, here's my ssn: 123-45-6789",
                "I'm doing great, here's my ssn: 123-45-6789",
                "I'm doing great, how about you?",
            ],
        }
    )

    schema = WorkflowMetricConfigBuilder().add(prompt_response_ssn_regex_module).build()

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.regex.ssn",
        "response",
        "response.regex.ssn",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.regex.ssn"] == 1
    assert actual["distribution/min"]["prompt.regex.ssn"] == 0
    assert actual["types/integral"]["prompt.regex.ssn"] == 4
    assert actual["distribution/max"]["response.regex.ssn"] == 1
    assert actual["distribution/min"]["response.regex.ssn"] == 0
    assert actual["types/integral"]["response.regex.ssn"] == 4


def test_prompt_regex_df_email_address():
    df = pd.DataFrame(
        {
            "prompt": [
                "This is a email address: foo@whylabs.ai",
                "This is a email address: foo@whylabs.ai",
                "This is a email address: foo@whylabs.ai",
                "hi",
            ],
            "response": [
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
            ],
        }
    )

    schema = WorkflowMetricConfigBuilder().add(prompt_email_address_regex_metric).build()

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.regex.email_address",
        "response",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.regex.email_address"] == 1
    assert actual["distribution/min"]["prompt.regex.email_address"] == 0
    assert actual["types/integral"]["prompt.regex.email_address"] == 4
    assert actual["types/string"]["prompt.regex.email_address"] == 0


def test_response_regex_df_email_address():
    df = pd.DataFrame(
        {
            "prompt": [
                "How are you?",
                "How are you?",
                "How are you?",
                "How are you?",
            ],
            "response": [
                "I'm doing great, here's my email address: foo@whylabs.ai",
                "I'm doing great, here's my email address: foo@whylabs.ai",
                "I'm doing great, here's my email address: foo@whylabs.ai",
                "I'm doing great, how about you?",
            ],
        }
    )

    schema = WorkflowMetricConfigBuilder().add(response_email_address_regex_metric).build()

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "response",
        "response.regex.email_address",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["response.regex.email_address"] == 1
    assert actual["distribution/min"]["response.regex.email_address"] == 0
    assert actual["types/integral"]["response.regex.email_address"] == 4


def test_prompt_response_df_email_address():
    df = pd.DataFrame(
        {
            "prompt": [
                "This is a email address: foo@whylabs.ai",
                "This is a email address: foo@whylabs.ai",
                "This is a email address: foo@whylabs.ai",
                "something else",
            ],
            "response": [
                "I'm doing great, here's my email address:foo@whylabs.ai",
                "I'm doing great, here's my email address:foo@whylabs.ai",
                "I'm doing great, here's my email address:foo@whylabs.ai",
                "something else",
            ],
        }
    )

    schema = WorkflowMetricConfigBuilder().add(prompt_response_email_address_regex_module).build()

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.regex.email_address",
        "response",
        "response.regex.email_address",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.regex.email_address"] == 1
    assert actual["distribution/min"]["prompt.regex.email_address"] == 0
    assert actual["types/integral"]["prompt.regex.email_address"] == 4
    assert actual["distribution/max"]["response.regex.email_address"] == 1
    assert actual["distribution/min"]["response.regex.email_address"] == 0
    assert actual["types/integral"]["response.regex.email_address"] == 4


def test_prompt_regex_df_phone_number():
    df = pd.DataFrame(
        {
            "prompt": [
                "This is a phone number: 123-456-7890",
                "This is a phone number: 123-456-7890",
                "This is a phone number: 123-456-7890",
                "This is a phone number: redacted",
            ],
            "response": [
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
            ],
        }
    )

    schema = WorkflowMetricConfigBuilder().add(prompt_phone_number_regex_metric).build()

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.regex.phone_number",
        "response",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.regex.phone_number"] == 1
    assert actual["distribution/min"]["prompt.regex.phone_number"] == 0
    assert actual["types/integral"]["prompt.regex.phone_number"] == 4


def test_response_regex_df_phone_number():
    df = pd.DataFrame(
        {
            "prompt": [
                "How are you?",
                "How are you?",
                "How are you?",
                "How are you?",
            ],
            "response": [
                "I'm doing great, here's my phone number: 123-456-7890",
                "I'm doing great, here's my phone number: 123-456-7890",
                "I'm doing great, here's my phone number: 123-456-7890",
                "I'm doing great, how about you?",
            ],
        }
    )

    schema = WorkflowMetricConfigBuilder().add(response_phone_number_regex_metric).build()

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "response",
        "response.regex.phone_number",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["response.regex.phone_number"] == 1
    assert actual["distribution/min"]["response.regex.phone_number"] == 0
    assert actual["types/integral"]["response.regex.phone_number"] == 4


def test_prompt_response_regex_df_phone_number():
    df = pd.DataFrame(
        {
            "prompt": [
                "This is a phone number: 123-456-7890",
                "This is a phone number: 123-456-7890",
                "This is a phone number: 123-456-7890",
                "How are you?",
            ],
            "response": [
                "I'm doing great, here's my phone number: 123-456-7890",
                "I'm doing great, here's my phone number: 123-456-7890",
                "I'm doing great, here's my phone number: 123-456-7890",
                "I'm doing great, how about you?",
            ],
        }
    )

    schema = WorkflowMetricConfigBuilder().add(prompt_response_phone_number_regex_module).build()

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.regex.phone_number",
        "response",
        "response.regex.phone_number",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.regex.phone_number"] == 1
    assert actual["distribution/mean"]["prompt.regex.phone_number"] == 0.75
    assert actual["distribution/min"]["prompt.regex.phone_number"] == 0
    assert actual["types/integral"]["prompt.regex.phone_number"] == 4
    assert actual["distribution/max"]["response.regex.phone_number"] == 1
    assert actual["distribution/mean"]["response.regex.phone_number"] == 0.75
    assert actual["distribution/min"]["response.regex.phone_number"] == 0
    assert actual["types/integral"]["response.regex.phone_number"] == 4


def test_prompt_regex_df_mailing_address():
    df = pd.DataFrame(
        {
            "prompt": [
                "This is a mailing address: 123 Main St, San Francisco, CA 94105",
                "This is a mailing address: 123 Main St, San Francisco, CA 94105",
                "This is a mailing address: 123 Main St, San Francisco, CA 94105",
                "This is a mailing address: redacted",
            ],
            "response": [
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
            ],
        }
    )

    schema = WorkflowMetricConfigBuilder().add(prompt_mailing_address_regex_metric).build()

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.regex.mailing_address",
        "response",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.regex.mailing_address"] == 1
    assert actual["distribution/min"]["prompt.regex.mailing_address"] == 0
    assert actual["types/integral"]["prompt.regex.mailing_address"] == 4


def test_response_regex_df_mailing_address():
    df = pd.DataFrame(
        {
            "prompt": [
                "How are you?",
                "How are you?",
                "How are you?",
                "How are you?",
            ],
            "response": [
                "I'm doing great, here's my mailing address: 123 Main St, San Francisco, CA 94105",
                "I'm doing great, here's my mailing address: 123 Main St, San Francisco, CA 94105",
                "I'm doing great, here's my mailing address: 123 Main St, San Francisco, CA 94105",
                "I'm doing great, how about you?",
            ],
        }
    )

    schema = WorkflowMetricConfigBuilder().add(response_mailing_address_regex_metric).build()

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "response",
        "response.regex.mailing_address",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["response.regex.mailing_address"] == 1
    assert actual["distribution/min"]["response.regex.mailing_address"] == 0
    assert actual["types/integral"]["response.regex.mailing_address"] == 4


def test_prompt_response_regex_df_mailing_address():
    df = pd.DataFrame(
        {
            "prompt": [
                "This is a mailing address: 123 Main St, San Francisco, CA 94105",
                "This is a mailing address: 123 Main St, San Francisco, CA 94105",
                "This is a mailing address: 123 Main St, San Francisco, CA 94105",
                "How are you?",
            ],
            "response": [
                "I'm doing great, here's my mailing address: 123 Main St, San Francisco, CA 94105",
                "I'm doing great, here's my mailing address: 123 Main St, San Francisco, CA 94105",
                "I'm doing great, here's my mailing address: 123 Main St, San Francisco, CA 94105",
                "I'm doing great, how about you?",
            ],
        }
    )

    schema = WorkflowMetricConfigBuilder().add(prompt_response_mailing_address_regex_module).build()

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.regex.mailing_address",
        "response",
        "response.regex.mailing_address",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.regex.mailing_address"] == 1
    assert actual["distribution/min"]["prompt.regex.mailing_address"] == 0
    assert actual["types/integral"]["prompt.regex.mailing_address"] == 4
    assert actual["distribution/max"]["response.regex.mailing_address"] == 1
    assert actual["distribution/min"]["response.regex.mailing_address"] == 0
    assert actual["types/integral"]["response.regex.mailing_address"] == 4


def test_prompt_regex_df_credit_card_number():
    df = pd.DataFrame(
        {
            "prompt": [
                "This is a credit card number: 1234-5678-9012-3456",
                "This is a credit card number: 1234-5678-9012-3456",
                "This is a credit card number: 1234-5678-9012-3456",
                "This is a credit card number: redacted",
            ],
            "response": [
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
            ],
        }
    )

    schema = WorkflowMetricConfigBuilder().add(prompt_credit_card_number_regex_metric).build()

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.regex.credit_card_number",
        "response",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.regex.credit_card_number"] == 1
    assert actual["distribution/min"]["prompt.regex.credit_card_number"] == 0
    assert actual["types/integral"]["prompt.regex.credit_card_number"] == 4


def test_response_regex_df_credit_card_number():
    df = pd.DataFrame(
        {
            "prompt": [
                "How are you?",
                "How are you?",
                "How are you?",
                "How are you?",
            ],
            "response": [
                "I'm doing great, here's my credit card number: 1234-5678-9012-3456",
                "I'm doing great, here's my credit card number: 1234-5678-9012-3456",
                "I'm doing great, here's my credit card number: 1234-5678-9012-3456",
                "I'm doing great, how about you?",
            ],
        }
    )

    schema = WorkflowMetricConfigBuilder().add(response_credit_card_number_regex_metric).build()

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "response",
        "response.regex.credit_card_number",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["response.regex.credit_card_number"] == 1
    assert actual["distribution/min"]["response.regex.credit_card_number"] == 0
    assert actual["types/integral"]["response.regex.credit_card_number"] == 4


def test_prompt_response_regex_df_credit_card_number():
    df = pd.DataFrame(
        {
            "prompt": [
                "This is a credit card number: 1234-5678-9012-3456",
                "This is a credit card number: 1234-5678-9012-3456",
                "This is a credit card number: 1234-5678-9012-3456",
                "How are you?",
            ],
            "response": [
                "I'm doing great, here's my credit card number: 1234-5678-9012-3456",
                "I'm doing great, here's my credit card number: 1234-5678-9012-3456",
                "I'm doing great, here's my credit card number: 1234-5678-9012-3456",
                "I'm doing great, how about you?",
            ],
        }
    )

    schema = WorkflowMetricConfigBuilder().add(prompt_response_credit_card_number_regex_module).build()

    actual = _log(df, schema)
    assert list(actual.columns) == expected_metrics

    expected_columns = [
        "prompt",
        "prompt.regex.credit_card_number",
        "response",
        "response.regex.credit_card_number",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.regex.credit_card_number"] == 1
    assert actual["distribution/min"]["prompt.regex.credit_card_number"] == 0
    assert actual["types/integral"]["prompt.regex.credit_card_number"] == 4
    assert actual["distribution/max"]["response.regex.credit_card_number"] == 1
    assert actual["distribution/min"]["response.regex.credit_card_number"] == 0
    assert actual["types/integral"]["response.regex.credit_card_number"] == 4


def test_prompt_regex_df_default():
    df = pd.DataFrame(
        {
            "prompt": [
                "This is a ssn: 123-45-6789",
                "This is a ssn: 123-45-6789",
                "This is a ssn: redacted",
                "This is a email address: foo@whylabs.ai",
            ],
            "response": [
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
                "I'm doing great, how about you?",
            ],
        }
    )

    schema = WorkflowMetricConfigBuilder().add(prompt_default_regexes_enum_metric).build()

    actual = _log(df, schema)
    expected = [
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
        "frequent_items/frequent_strings",
        "type",
        "types/boolean",
        "types/fractional",
        "types/integral",
        "types/object",
        "types/string",
        "types/tensor",
    ]

    assert sorted(list(actual.columns)) == sorted(expected)  # type: ignore

    expected_columns = [
        "prompt",
        "prompt.regex.has_patterns",
        "response",
    ]

    # FI shouldn't be used for prompt/response, only the has_patterns
    assert math.isnan(actual["frequent_items/frequent_strings"]["prompt"])  # type: ignore
    assert math.isnan(actual["frequent_items/frequent_strings"]["response"])  # type: ignore

    assert actual.index.tolist() == expected_columns
    assert actual["frequent_items/frequent_strings"]["prompt.regex.has_patterns"] == [
        FrequentItem(value="SSN", est=2, upper=2, lower=2),
        FrequentItem(value="EMAIL_ADDRESS", est=1, upper=1, lower=1),
    ]


def test_response_regex_df_default():
    df = pd.DataFrame(
        {
            "prompt": [
                "How are you?",
                "How are you?",
                "How are you?",
                "How are you?",
            ],
            "response": [
                "I'm doing great, here's my phone number: 123-456-7890",
                "I'm doing great, here's my ssn: 123-45-6789",
                "I'm doing great, here's my ssn: 123-45-6789",
                "I'm doing great, here's my email address: foo@whylabs.ai",
            ],
        }
    )

    schema = WorkflowMetricConfigBuilder().add(response_default_regexes_module).build()

    actual = _log(df, schema)

    expected = [
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
        "frequent_items/frequent_strings",
        "type",
        "types/boolean",
        "types/fractional",
        "types/integral",
        "types/object",
        "types/string",
        "types/tensor",
    ]

    assert sorted(list(actual.columns)) == sorted(expected)  # type: ignore

    expected_columns = [
        "prompt",
        "response",
        "response.regex.has_patterns",
    ]

    # FI shouldn't be used for prompt/response, only the has_patterns
    assert math.isnan(actual["frequent_items/frequent_strings"]["prompt"])  # type: ignore
    assert math.isnan(actual["frequent_items/frequent_strings"]["response"])  # type: ignore

    assert actual.index.tolist() == expected_columns
    assert actual["frequent_items/frequent_strings"]["response.regex.has_patterns"] == [
        FrequentItem(value="SSN", est=2, upper=2, lower=2),
        FrequentItem(value="EMAIL_ADDRESS", est=1, upper=1, lower=1),
        FrequentItem(value="PHONE_NUMBER", est=1, upper=1, lower=1),
    ]


def test_prompt_response_regex_df_default():
    df = pd.DataFrame(
        {
            "prompt": [
                "This is a ssn: 123-45-6789",
                "This is a ssn: 123-45-6789",
                "This is a ssn: redacted",
                "This is a email address: foo@whylabs.ai",
            ],
            "response": [
                "I'm doing great, here's my phone number: 123-456-7890",
                "I'm doing great, here's my ssn: 123-45-6789",
                "I'm doing great, here's my ssn: 123-45-6789",
                "I'm doing great, here's my email address: foo@whylabs.ai",
            ],
        }
    )

    schema = WorkflowMetricConfigBuilder().add(prompt_response_default_regexes_module).build()

    actual = _log(df, schema)
    expected = [
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
        "frequent_items/frequent_strings",
        "type",
        "types/boolean",
        "types/fractional",
        "types/integral",
        "types/object",
        "types/string",
        "types/tensor",
    ]

    assert sorted(list(actual.columns)) == sorted(expected)  # type: ignore

    expected_columns = [
        "prompt",
        "prompt.regex.has_patterns",
        "response",
        "response.regex.has_patterns",
    ]

    # FI shouldn't be used for prompt/response, only the has_patterns
    assert math.isnan(actual["frequent_items/frequent_strings"]["prompt"])  # type: ignore
    assert math.isnan(actual["frequent_items/frequent_strings"]["response"])  # type: ignore

    assert actual.index.tolist() == expected_columns
    assert actual["frequent_items/frequent_strings"]["prompt.regex.has_patterns"] == [
        FrequentItem(value="SSN", est=2, upper=2, lower=2),
        FrequentItem(value="EMAIL_ADDRESS", est=1, upper=1, lower=1),
    ]
    assert actual["frequent_items/frequent_strings"]["response.regex.has_patterns"] == [
        FrequentItem(value="SSN", est=2, upper=2, lower=2),
        FrequentItem(value="EMAIL_ADDRESS", est=1, upper=1, lower=1),
        FrequentItem(value="PHONE_NUMBER", est=1, upper=1, lower=1),
    ]


def test_prompt_response_ssn_phone_number():
    df = pd.DataFrame(
        {
            "prompt": [
                "This is a ssn: 123-45-6789",
                "This is a ssn: 123-45-6789",
                "This is a ssn: redacted",
                "This is a phone number: 123-456-7890",
            ],
            "response": [
                "I'm doing great, here's my phone number: 123-456-7890",
                "I'm doing great, here's my ssn: 123-45-6789",
                "I'm doing great, here's my ssn: 123-45-6789",
                "I'm doing great, here's my email address: foo@gmail.com",
            ],
        }
    )

    # mix and match several different ones
    schema = (
        WorkflowMetricConfigBuilder()
        .add(prompt_response_ssn_regex_module + prompt_response_phone_number_regex_module + prompt_response_default_regexes_module)
        .build()
    )

    actual = _log(df, schema)
    expected_columns = [
        "prompt",
        "prompt.regex.has_patterns",
        "prompt.regex.phone_number",
        "prompt.regex.ssn",
        "response",
        "response.regex.has_patterns",
        "response.regex.phone_number",
        "response.regex.ssn",
    ]

    assert actual.index.tolist() == expected_columns

    assert actual["distribution/max"]["prompt.regex.phone_number"] == 1
    assert actual["distribution/min"]["prompt.regex.phone_number"] == 0
    assert actual["types/integral"]["prompt.regex.phone_number"] == 4
    assert actual["distribution/max"]["prompt.regex.ssn"] == 1
    assert actual["distribution/min"]["prompt.regex.ssn"] == 0
    assert actual["types/integral"]["prompt.regex.ssn"] == 4

    assert actual["distribution/max"]["response.regex.phone_number"] == 1
    assert actual["distribution/min"]["response.regex.phone_number"] == 0
    assert actual["types/integral"]["response.regex.phone_number"] == 4
    assert actual["distribution/max"]["response.regex.ssn"] == 1
    assert actual["distribution/min"]["response.regex.ssn"] == 0
    assert actual["types/integral"]["response.regex.ssn"] == 4

    assert actual["frequent_items/frequent_strings"]["prompt.regex.has_patterns"] == [
        FrequentItem(value="SSN", est=2, upper=2, lower=2),
        FrequentItem(value="PHONE_NUMBER", est=1, upper=1, lower=1),
    ]
    assert actual["frequent_items/frequent_strings"]["response.regex.has_patterns"] == [
        FrequentItem(value="SSN", est=2, upper=2, lower=2),
        FrequentItem(value="EMAIL_ADDRESS", est=1, upper=1, lower=1),
        FrequentItem(value="PHONE_NUMBER", est=1, upper=1, lower=1),
    ]


def test_custom_regex_frequent_item():
    df = pd.DataFrame(
        {
            "prompt": [
                "This is a ssn: 123-45-6789",
                "This is a ssn: 123-45-6789",
                "This is a ssn: redacted",
                "This is a phone number: 123-456-7890",
                "This is my password: password123",
            ],
            "response": [
                "I'm doing great, here's my phone number: 123-456-7890",
                "I'm doing great, here's my ssn: 123-45-6789",
                "I'm doing great, here's my ssn: 123-45-6789",
                "I'm doing great, here's my email address: foo@gmail.com",
                "Nice password!",
            ],
        }
    )

    regexes: PatternGroups = {"patterns": [{"name": "password detector", "expressions": [r"\bpassword\b"], "substitutions": []}]}

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        json.dump(regexes["patterns"], f)

    custom_regex_modules = get_custom_regex_frequent_items_modules(f.name)
    schema = WorkflowMetricConfigBuilder().add(custom_regex_modules.prompt_custom_regexes_frequent_items_module).build()

    actual = _log(df, schema)

    expected_columns = [
        "prompt",
        "prompt.regex.has_patterns",
        "response",
    ]

    assert actual.index.tolist() == expected_columns

    assert actual["frequent_items/frequent_strings"]["prompt.regex.has_patterns"] == [
        FrequentItem(value="PASSWORD_DETECTOR", est=1, upper=1, lower=1),
    ]

    os.remove(f.name)


def test_custom_regex():
    df = pd.DataFrame(
        {
            "prompt": [
                "This is a ssn: 123-45-6789",
                "This is a ssn: 123-45-6789",
                "This is a ssn: redacted",
                "This is a phone number: 123-456-7890",
                "This is my password: password123",
            ],
            "response": [
                "foo I'm doing great, here's my phone number: 123-456-7890",
                "foo I'm doing great, here's my ssn: 123-45-6789",
                "foo I'm doing great, here's my ssn: 123-45-6789",
                "foo I'm doing great, here's my email address: foo@gmail.com",
                "Nice foo!",
            ],
        }
    )

    password_detector: CompiledPatternGroups = {
        "patterns": [
            {"name": "password detector", "expressions": [re.compile(r"\bpassword\b")], "substitutions": []},
        ]
    }

    foo_detector: CompiledPatternGroups = {
        "patterns": [
            {"name": "foo detector", "expressions": [re.compile(r"\bfoo\b")], "substitutions": []},
        ]
    }

    prompt_modules = get_custom_regex_modules(password_detector)
    response_modules = get_custom_regex_modules(foo_detector)
    schema = (
        WorkflowMetricConfigBuilder()
        .add(prompt_modules.prompt_custom_regex_module)
        .add(response_modules.response_custom_regex_module)
        .build()
    )

    actual = _log(df, schema)

    expected_columns = [
        "prompt",
        "prompt.regex.password_detector",
        "response",
        "response.regex.foo_detector",
    ]

    assert actual.index.tolist() == expected_columns
    assert actual["distribution/max"]["prompt.regex.password_detector"] == 1
    assert actual["distribution/min"]["prompt.regex.password_detector"] == 0
    assert actual["distribution/max"]["response.regex.foo_detector"] == 1
    assert actual["distribution/min"]["response.regex.foo_detector"] == 1


def test_custom_regex_custom_columns():
    df = pd.DataFrame(
        {
            "my_prompt": [
                "This is a ssn: 123-45-6789",
                "This is a ssn: 123-45-6789",
                "This is a ssn: redacted",
                "This is a phone number: 123-456-7890",
                "This is my password: password123",
            ],
            "my_response": [
                "foo I'm doing great, here's my phone number: 123-456-7890",
                "foo I'm doing great, here's my ssn: 123-45-6789",
                "foo I'm doing great, here's my ssn: 123-45-6789",
                "foo I'm doing great, here's my email address: foo@gmail.com",
                "Nice foo!",
            ],
        }
    )

    password_detector: CompiledPatternGroups = {
        "patterns": [
            {"name": "password detector", "expressions": [re.compile(r"\bpassword\b")], "substitutions": []},
        ]
    }

    foo_detector: CompiledPatternGroups = {
        "patterns": [
            {"name": "foo detector", "expressions": [re.compile(r"\bfoo\b")], "substitutions": []},
        ]
    }

    prompt_module = get_custom_regex_frequent_items_for_column_module("my_prompt", password_detector)
    response_module = get_custom_regex_frequent_items_for_column_module("my_response", foo_detector)
    schema = WorkflowMetricConfigBuilder().add(prompt_module).add(response_module).build()

    actual = _log(df, schema)

    expected_columns = [
        "my_prompt",
        "my_prompt.regex.has_patterns",
        "my_response",
        "my_response.regex.has_patterns",
    ]

    assert actual.index.tolist() == expected_columns
    pd.set_option("display.max_columns", None)
    assert actual["frequent_items/frequent_strings"]["my_prompt.regex.has_patterns"] == [
        FrequentItem(value="PASSWORD_DETECTOR", est=1, upper=1, lower=1),
    ]
    assert actual["frequent_items/frequent_strings"]["my_response.regex.has_patterns"] == [
        FrequentItem(value="FOO_DETECTOR", est=5, upper=5, lower=5),
    ]
