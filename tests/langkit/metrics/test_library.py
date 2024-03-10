from typing import List

from langkit.core.workflow import EvaluationWorkflow
from langkit.metrics.library import lib


def test_recommended():
    wf = EvaluationWorkflow(metrics=[lib.presets.recommended()])
    result = wf.run({"prompt": "hi", "response": "I'm doing great!"})
    metrics = result.metrics

    metric_names: List[str] = metrics.columns.tolist()  # pyright: ignore[reportUnknownMemberType]

    assert metric_names == [
        "prompt.pii.phone_number",
        "prompt.pii.email_address",
        "prompt.pii.credit_card",
        "prompt.pii.us_ssn",
        "prompt.pii.us_bank_number",
        "prompt.pii.redacted",
        "prompt.stats.token_count",
        "prompt.stats.char_count",
        "prompt.similarity.injection",
        "prompt.similarity.jailbreak",
        "response.pii.phone_number",
        "response.pii.email_address",
        "response.pii.credit_card",
        "response.pii.us_ssn",
        "response.pii.us_bank_number",
        "response.pii.redacted",
        "response.stats.token_count",
        "response.stats.char_count",
        "response.stats.flesch_reading_ease",
        "response.sentiment.sentiment_score",
        "response.toxicity.toxicity_score",
        "response.similarity.refusal",
        "id",
    ]
