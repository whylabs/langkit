# pyright: reportUnknownMemberType=none
import pandas as pd

import whylogs as why
from langkit.core.metric import EvaluationConfig, EvaluationConfigBuilder
from langkit.core.workflow import EvaluationWorkflow
from langkit.metrics.pii import prompt_response_presidio_pii_metric
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


def _log(item: pd.DataFrame, conf: EvaluationConfig) -> pd.DataFrame:
    schema = create_whylogs_udf_schema(conf)
    return why.log(item, schema=schema).view().to_pandas()  # type: ignore


def test_prompt_response_pii_metric_whylogs():
    all_config = EvaluationConfigBuilder().add(prompt_response_presidio_pii_metric).build()

    df = pd.DataFrame(
        {
            "prompt": [
                "Hey! Here is my phone number: 555-555-5555, and my email is foo@bar.com.",
            ],
            "response": [
                "That's a cool phone number! Checkout google.com",
            ],
        }
    )

    expected_rows = [
        "prompt",
        "prompt.pii.credit_card",
        "prompt.pii.email_address",
        "prompt.pii.ip_address",
        "prompt.pii.phone_number",
        "prompt.pii.redacted",
        "prompt.pii.url",
        "response",
        "response.pii.credit_card",
        "response.pii.email_address",
        "response.pii.ip_address",
        "response.pii.phone_number",
        "response.pii.redacted",
        "response.pii.url",
    ]

    logged = _log(item=df, conf=all_config)

    assert list(logged.columns) == expected_metrics
    assert logged.index.tolist() == expected_rows

    assert logged["distribution/max"]["prompt.pii.phone_number"] == 1
    assert logged["distribution/max"]["prompt.pii.email_address"] == 1
    assert logged["distribution/max"]["prompt.pii.credit_card"] == 0.0
    assert logged["distribution/max"]["prompt.pii.url"] == 1.0  # the email triggers this too
    assert logged["distribution/max"]["prompt.pii.ip_address"] == 0.0
    assert logged["distribution/max"]["response.pii.phone_number"] == 0.0
    assert logged["distribution/max"]["response.pii.email_address"] == 0.0
    assert logged["distribution/max"]["response.pii.credit_card"] == 0.0
    assert logged["distribution/max"]["response.pii.url"] == 1.0
    assert logged["distribution/max"]["response.pii.ip_address"] == 0.0


def test_prompt_response_pii_metric():
    df = pd.DataFrame(
        {
            "prompt": [
                "Hey! Here is my phone number: 555-555-5555, and my email is foo@bar.com.",
            ],
            "response": [
                "That's a cool phone number! Checkout google.com",
            ],
        }
    )

    expected_rows = [
        "prompt.pii.phone_number",
        "prompt.pii.email_address",
        "prompt.pii.credit_card",
        "prompt.pii.ip_address",
        "prompt.pii.redacted",
        "response.pii.phone_number",
        "response.pii.email_address",
        "response.pii.credit_card",
        "response.pii.ip_address",
        "response.pii.redacted",
        "id",
    ]

    wf = EvaluationWorkflow(metrics=[prompt_response_presidio_pii_metric])
    logged = wf.run(df).metrics

    pd.set_option("display.max_columns", None)
    print(logged.transpose())

    assert list(logged.columns) == expected_rows

    assert logged["prompt.pii.phone_number"][0] == 1
    assert logged["prompt.pii.email_address"][0] == 1
    assert logged["prompt.pii.credit_card"][0] == 0
    assert logged["prompt.pii.ip_address"][0] == 0
    assert logged["response.pii.phone_number"][0] == 0
    assert logged["response.pii.email_address"][0] == 0
    assert logged["response.pii.credit_card"][0] == 0
    assert logged["response.pii.ip_address"][0] == 0
    assert logged["prompt.pii.redacted"][0] == "Hey! Here is my phone number: <PHONE_NUMBER>, and my email is <EMAIL_ADDRESS>."
    assert logged["response.pii.redacted"][0] is None
    assert logged["id"][0] == 0
