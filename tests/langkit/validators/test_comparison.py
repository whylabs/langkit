import pytest

from langkit.core.validation import ValidationFailure
from langkit.core.workflow import EvaluationWorkflow
from langkit.metrics.library import lib as metric_lib
from langkit.validators.comparison import ConstraintValidator


def test_one_required():
    with pytest.raises(Exception):
        ConstraintValidator("prompt.text_stat.char_count")


def test_upper_threshold():
    validator = ConstraintValidator("prompt.text_stat.char_count", upper_threshold=5)
    wf = EvaluationWorkflow(metrics=[metric_lib.prompt.text_stat.char_count], validators=[validator])

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


def test_lower_threshold():
    validator = ConstraintValidator("prompt.text_stat.char_count", lower_threshold=5)
    wf = EvaluationWorkflow(metrics=[metric_lib.prompt.text_stat.char_count], validators=[validator])

    result = wf.run({"prompt": "1"})

    assert result.validation_results.report == [
        ValidationFailure(
            id="0",
            metric="prompt.text_stat.char_count",
            details="Value 1 is below threshold 5",
            value=1,
            lower_threshold=5,
        )
    ]


def test_upper_threshold_inclusive():
    validator = ConstraintValidator("prompt.text_stat.char_count", upper_threshold_inclusive=5)
    wf = EvaluationWorkflow(metrics=[metric_lib.prompt.text_stat.char_count], validators=[validator])

    result = wf.run({"prompt": "12345"})

    assert result.validation_results.report == [
        ValidationFailure(
            id="0",
            metric="prompt.text_stat.char_count",
            details="Value 5 is above or equal to threshold 5",
            value=5,
            upper_threshold=5,
        )
    ]


def test_lower_threshold_inclusive():
    validator = ConstraintValidator("prompt.text_stat.char_count", lower_threshold_inclusive=5)
    wf = EvaluationWorkflow(metrics=[metric_lib.prompt.text_stat.char_count], validators=[validator])

    result = wf.run({"prompt": "12345"})

    assert result.validation_results.report == [
        ValidationFailure(
            id="0",
            metric="prompt.text_stat.char_count",
            details="Value 5 is below or equal to threshold 5",
            value=5,
            lower_threshold=5,
        )
    ]


def test_one_of():
    validator = ConstraintValidator("prompt.text_stat.char_count", one_of=[1, 2, 3])
    wf = EvaluationWorkflow(metrics=[metric_lib.prompt.text_stat.char_count], validators=[validator])

    result = wf.run({"prompt": "asdf"})

    assert result.validation_results.report == [
        ValidationFailure(
            id="0",
            metric="prompt.text_stat.char_count",
            details="Value 4 is not in allowed values {1, 2, 3}",
            value=4,
            allowed_values=[1, 2, 3],
        )
    ]


def test_none_of():
    validator = ConstraintValidator("prompt.text_stat.char_count", none_of=[1, 2, 3])
    wf = EvaluationWorkflow(metrics=[metric_lib.prompt.text_stat.char_count], validators=[validator])

    result = wf.run({"prompt": "asd"})

    assert result.validation_results.report == [
        ValidationFailure(
            id="0",
            metric="prompt.text_stat.char_count",
            details="Value 3 is in disallowed values {1, 2, 3}",
            value=3,
            disallowed_values=[1, 2, 3],
        )
    ]


def test_must_be_none():
    validator = ConstraintValidator("prompt.pii.redacted", must_be_none=True)
    wf = EvaluationWorkflow(metrics=[metric_lib.prompt.pii()], validators=[validator])

    result = wf.run({"prompt": "My email address is anthony@whylabs.ai"})

    assert result.validation_results.report == [
        ValidationFailure(
            id="0",
            metric="prompt.pii.redacted",
            details="Value My email address is <EMAIL_ADDRESS> is not None",
            value="My email address is <EMAIL_ADDRESS>",
            must_be_none=True,
        ),
    ]


def test_must_be_non_none():
    validator = ConstraintValidator("prompt.pii.redacted", must_be_non_none=True)
    wf = EvaluationWorkflow(metrics=[metric_lib.prompt.pii()], validators=[validator])

    result = wf.run({"prompt": "My email address is not here"})

    assert result.validation_results.report == [
        ValidationFailure(
            id="0",
            metric="prompt.pii.redacted",
            details="Value is None",
            value=None,
            must_be_non_none=True,
        ),
    ]
