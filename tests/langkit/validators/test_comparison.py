import pytest

from langkit.core.validation import ValidationFailure
from langkit.core.workflow import Workflow
from langkit.metrics.library import lib as metric_lib
from langkit.validators.comparison import (
    ConstraintValidator,
    ConstraintValidatorOptions,
    MultiColumnConstraintValidator,
    MultiColumnConstraintValidatorOptions,
)


def test_one_required():
    with pytest.raises(Exception):
        ConstraintValidator(ConstraintValidatorOptions("prompt.stats.char_count"))


def test_upper_threshold():
    validator = ConstraintValidator(ConstraintValidatorOptions("prompt.stats.char_count", upper_threshold=5))
    wf = Workflow(metrics=[metric_lib.prompt.stats.char_count], validators=[validator])

    result = wf.run({"prompt": "1234567890"})

    assert result.validation_results.report == [
        ValidationFailure(
            id="0",
            metric="prompt.stats.char_count",
            details="Value 10 is above threshold 5",
            value=10,
            upper_threshold=5,
        )
    ]


def test_lower_threshold():
    validator = ConstraintValidator(ConstraintValidatorOptions("prompt.stats.char_count", lower_threshold=5))
    wf = Workflow(metrics=[metric_lib.prompt.stats.char_count], validators=[validator])

    result = wf.run({"prompt": "1"})

    assert result.validation_results.report == [
        ValidationFailure(
            id="0",
            metric="prompt.stats.char_count",
            details="Value 1 is below threshold 5",
            value=1,
            lower_threshold=5,
        )
    ]


def test_upper_threshold_inclusive():
    validator = ConstraintValidator(ConstraintValidatorOptions("prompt.stats.char_count", upper_threshold_inclusive=5))
    wf = Workflow(metrics=[metric_lib.prompt.stats.char_count], validators=[validator])

    result = wf.run({"prompt": "12345"})

    assert result.validation_results.report == [
        ValidationFailure(
            id="0",
            metric="prompt.stats.char_count",
            details="Value 5 is above or equal to threshold 5",
            value=5,
            upper_threshold=5,
        )
    ]


def test_lower_threshold_inclusive():
    validator = ConstraintValidator(ConstraintValidatorOptions("prompt.stats.char_count", lower_threshold_inclusive=5))
    wf = Workflow(metrics=[metric_lib.prompt.stats.char_count], validators=[validator])

    result = wf.run({"prompt": "12345"})

    assert result.validation_results.report == [
        ValidationFailure(
            id="0",
            metric="prompt.stats.char_count",
            details="Value 5 is below or equal to threshold 5",
            value=5,
            lower_threshold=5,
        )
    ]


def test_one_of():
    validator = ConstraintValidator(ConstraintValidatorOptions("prompt.stats.char_count", one_of=[1, 2, 3]))
    wf = Workflow(metrics=[metric_lib.prompt.stats.char_count], validators=[validator])

    result = wf.run({"prompt": "asdf"})

    assert result.validation_results.report == [
        ValidationFailure(
            id="0",
            metric="prompt.stats.char_count",
            details="Value 4 is not in allowed values {1, 2, 3}",
            value=4,
            allowed_values=[1, 2, 3],
        )
    ]


def test_none_of():
    validator = ConstraintValidator(ConstraintValidatorOptions("prompt.stats.char_count", none_of=[1, 2, 3]))
    wf = Workflow(metrics=[metric_lib.prompt.stats.char_count], validators=[validator])

    result = wf.run({"prompt": "asd"})

    assert result.validation_results.report == [
        ValidationFailure(
            id="0",
            metric="prompt.stats.char_count",
            details="Value 3 is in disallowed values {1, 2, 3}",
            value=3,
            disallowed_values=[1, 2, 3],
        )
    ]


def test_must_be_none():
    validator = ConstraintValidator(ConstraintValidatorOptions("prompt.pii.redacted", must_be_none=True))
    wf = Workflow(metrics=[metric_lib.prompt.pii()], validators=[validator])

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
    validator = ConstraintValidator(ConstraintValidatorOptions("prompt.pii.redacted", must_be_non_none=True))
    wf = Workflow(metrics=[metric_lib.prompt.pii()], validators=[validator])

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


def test_multiple_contraint_first_failure():
    validator = MultiColumnConstraintValidator(
        MultiColumnConstraintValidatorOptions(
            [
                ConstraintValidatorOptions("prompt.stats.char_count", lower_threshold=5),
                ConstraintValidatorOptions("prompt.stats.token_count", lower_threshold=5),
            ]
        )
    )
    wf = Workflow(
        metrics=[metric_lib.prompt.stats.char_count, metric_lib.prompt.stats.token_count],
        validators=[validator],
    )

    result = wf.run({"prompt": "hi"})

    assert result.validation_results.report == [
        ValidationFailure(
            id="0",
            metric="prompt.stats.char_count",
            details="Value 2 is below threshold 5. "
            "Triggered because of failures in prompt.stats.char_count, prompt.stats.token_count (AND).",
            value=2,
            upper_threshold=None,
            lower_threshold=5,
            allowed_values=None,
            disallowed_values=None,
            must_be_none=None,
            must_be_non_none=None,
        ),
    ]


def test_multiple_constriant_all_failure():
    validator = MultiColumnConstraintValidator(
        MultiColumnConstraintValidatorOptions(
            [
                ConstraintValidatorOptions("prompt.stats.char_count", lower_threshold=5),
                ConstraintValidatorOptions("prompt.stats.token_count", lower_threshold=5),
            ],
            report_mode="ALL_FAILED_METRICS",
            operator="AND",
        )
    )
    wf = Workflow(
        metrics=[metric_lib.prompt.stats.char_count, metric_lib.prompt.stats.token_count],
        validators=[validator],
    )

    result = wf.run({"prompt": "hi hi"})

    assert result.validation_results.report == [
        ValidationFailure(
            id="0",
            metric="prompt.stats.char_count",
            details="Value 4 is below threshold 5",
            value=4,
            upper_threshold=None,
            lower_threshold=5,
            allowed_values=None,
            disallowed_values=None,
            must_be_none=None,
            must_be_non_none=None,
        ),
        ValidationFailure(
            id="0",
            metric="prompt.stats.token_count",
            details="Value 2 is below threshold 5",
            value=2,
            upper_threshold=None,
            lower_threshold=5,
            allowed_values=None,
            disallowed_values=None,
            must_be_none=None,
            must_be_non_none=None,
        ),
    ]


def test_multiple_constriant_all_failure_or():
    validator = MultiColumnConstraintValidator(
        MultiColumnConstraintValidatorOptions(
            [
                ConstraintValidatorOptions("prompt.stats.char_count", lower_threshold=4),
                ConstraintValidatorOptions("prompt.stats.token_count", lower_threshold=4),
            ],
            report_mode="ALL_FAILED_METRICS",
            operator="OR",
        )
    )
    wf = Workflow(
        metrics=[metric_lib.prompt.stats.char_count, metric_lib.prompt.stats.token_count],
        validators=[validator],
    )

    result = wf.run({"prompt": "hi hi"})

    assert result.validation_results.report == [
        ValidationFailure(
            id="0",
            metric="prompt.stats.token_count",
            details="Value 2 is below threshold 4",
            value=2,
            upper_threshold=None,
            lower_threshold=4,
            allowed_values=None,
            disallowed_values=None,
            must_be_none=None,
            must_be_non_none=None,
        ),
    ]


def test_multiple_constriant_first_failure_or():
    validator = MultiColumnConstraintValidator(
        MultiColumnConstraintValidatorOptions(
            [
                ConstraintValidatorOptions("prompt.stats.char_count", lower_threshold=4),
                ConstraintValidatorOptions("prompt.stats.token_count", lower_threshold=4),
            ],
            report_mode="FIRST_FAILED_METRIC",
            operator="OR",
        )
    )
    wf = Workflow(
        metrics=[metric_lib.prompt.stats.char_count, metric_lib.prompt.stats.token_count],
        validators=[validator],
    )

    result = wf.run({"prompt": "hi hi"})

    assert result.validation_results.report == [
        ValidationFailure(
            id="0",
            metric="prompt.stats.token_count",
            details="Value 2 is below threshold 4",
            value=2,
            upper_threshold=None,
            lower_threshold=4,
            allowed_values=None,
            disallowed_values=None,
            must_be_none=None,
            must_be_non_none=None,
        ),
    ]


def test_multiple_constriant_first_failure_or_multiple_failures():
    validator = MultiColumnConstraintValidator(
        MultiColumnConstraintValidatorOptions(
            [
                ConstraintValidatorOptions("prompt.stats.char_count", upper_threshold=100),
                ConstraintValidatorOptions("prompt.stats.token_count", upper_threshold=1),
                ConstraintValidatorOptions("prompt.regex.email_address", upper_threshold=0),
            ],
            report_mode="FIRST_FAILED_METRIC",
            operator="OR",
        )
    )
    wf = Workflow(
        metrics=[metric_lib.prompt.stats.char_count, metric_lib.prompt.stats.token_count, metric_lib.prompt.regex.email_address],
        validators=[validator],
    )

    result = wf.run({"prompt": "hi foo@whylabs.ai"})

    assert result.validation_results.report == [
        ValidationFailure(
            id="0",
            metric="prompt.stats.token_count",
            details="Value 7 is above threshold 1. "
            "Triggered because of failures in prompt.stats.token_count, prompt.regex.email_address (OR).",
            value=7,
            upper_threshold=1,
            lower_threshold=None,
            allowed_values=None,
            disallowed_values=None,
            must_be_none=None,
            must_be_non_none=None,
        ),
    ]


def test_multiple_constriant_first_failure_and_ordering():
    validator = MultiColumnConstraintValidator(
        MultiColumnConstraintValidatorOptions(
            [
                ConstraintValidatorOptions("prompt.regex.email_address", upper_threshold=0),
                ConstraintValidatorOptions("prompt.stats.char_count", upper_threshold=1),
                ConstraintValidatorOptions("prompt.stats.token_count", upper_threshold=1),
            ],
            report_mode="FIRST_FAILED_METRIC",
            operator="AND",
        )
    )
    wf = Workflow(
        metrics=[metric_lib.prompt.stats.char_count, metric_lib.prompt.stats.token_count, metric_lib.prompt.regex.email_address],
        validators=[validator],
    )

    result = wf.run({"prompt": "hi foo@whylabs.ai"})

    assert result.validation_results.report == [
        ValidationFailure(
            id="0",
            metric="prompt.regex.email_address",
            details="Value 1 is above threshold 0. "
            "Triggered because of failures in prompt.regex.email_address, prompt.stats.char_count, prompt.stats.token_count (AND).",
            value=1,
            upper_threshold=0,
            lower_threshold=None,
            allowed_values=None,
            disallowed_values=None,
            must_be_none=None,
            must_be_non_none=None,
        ),
    ]
