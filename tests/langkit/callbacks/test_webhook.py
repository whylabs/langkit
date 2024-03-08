from langkit.callbacks.library import lib as callback_lib
from langkit.core.validation import ValidationFailure
from langkit.core.workflow import EvaluationWorkflow
from langkit.metrics.library import lib as metric_lib
from langkit.validators.library import lib as validator_lib


def test_webhook_failures_dont_ruin_run():
    wf = EvaluationWorkflow(
        metrics=[metric_lib.prompt.text_stat.char_count],
        validators=[validator_lib.constraint("prompt.text_stat.char_count", upper_threshold=5)],
        callbacks=[callback_lib.webhook.basic_validation_failure("https://foo.bar")],  # will fail, url doesn't exist
    )

    result = wf.run({"prompt": "1234567890"})

    assert result.validation_results.report == [
        ValidationFailure(
            id="0",
            metric="prompt.text_stat.char_count",
            details="Value 10 is above threshold 5",
            value=10,
            upper_threshold=5,
        )
    ]
    assert result.metrics["prompt.text_stat.char_count"][0] == 10