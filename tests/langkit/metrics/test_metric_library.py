from langkit.core.workflow import EvaluationWorkflow
from langkit.metrics.library import lib


def test_recommended():
    row = {"prompt": "Hi, how are you doing today?", "response": "I'm doing great, how about you?"}
    wf = EvaluationWorkflow(metrics=[lib.presets.recommended()])

    actual = wf.run(row)

    expected_columns = [
        "prompt.pii.phone_number",
        "prompt.pii.email_address",
        "prompt.pii.credit_card",
        "prompt.pii.us_ssn",
        "prompt.pii.us_bank_number",
        "prompt.pii.redacted",
        "prompt.stats.token_count",
        "prompt.text_stat.char_count",
        "prompt.similarity.injection",
        "prompt.similarity.jailbreak",
        "response.pii.phone_number",
        "response.pii.email_address",
        "response.pii.credit_card",
        "response.pii.us_ssn",
        "response.pii.us_bank_number",
        "response.pii.redacted",
        "response.stats.token_count",
        "response.text_stat.char_count",
        "response.text_stat.flesch_reading_ease",
        "response.sentiment.sentiment_score",
        "response.toxicity",
        "response.similarity.refusal",
        "id",
    ]

    assert list(actual.metrics.columns) == expected_columns
