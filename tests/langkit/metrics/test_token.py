from langkit.core.workflow import EvaluationWorkflow
from langkit.metrics.library import lib


def test_token_count():
    row = {"prompt": "Hi, how are you doing today?", "response": "I'm doing great, how about you?"}
    wf = EvaluationWorkflow(metrics=[lib.prompt.stats.token_count(), lib.response.stats.token_count()])

    actual = wf.run(row)

    expected_columns = [
        "prompt.stats.token_count",
        "response.stats.token_count",
        "id",
    ]

    assert list(actual.metrics.columns) == expected_columns
    assert actual.metrics["prompt.stats.token_count"].tolist() == [8]
    assert actual.metrics["response.stats.token_count"].tolist() == [9]
