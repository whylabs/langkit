# pyright: reportUnknownMemberType=none
import pandas as pd

import whylogs as why
from langkit.core.metric import EvaluationConfig, EvaluationConfigBuilder
from langkit.metrics.pii import prompt_response_presidio_pii_module
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


def test_prompt_response_themes_module():
    all_config = EvaluationConfigBuilder().add(prompt_response_presidio_pii_module).build()

    df = pd.DataFrame(
        {
            "prompt": [
                "Hey! Here is my phone number: 555-555-5555, and my email is foo@bar.com.",
            ],
            "response": [
                "That's a cool phone number!",
            ],
        }
    )

    expected_rows = [
        "prompt",
        "prompt.pii.anonymized",
        "prompt.pii.credit_card",
        "prompt.pii.email_address",
        "prompt.pii.phone_number",
        "response",
        "response.pii.anonymized",
        "response.pii.credit_card",
        "response.pii.email_address",
        "response.pii.phone_number",
    ]

    logged = _log(item=df, conf=all_config)

    assert list(logged.columns) == expected_metrics
    assert logged.index.tolist() == expected_rows

    assert logged["distribution/max"]["prompt.pii.phone_number"] == 1
    assert logged["distribution/max"]["prompt.pii.email_address"] == 1
    assert logged["distribution/max"]["prompt.pii.credit_card"] == 0.0
