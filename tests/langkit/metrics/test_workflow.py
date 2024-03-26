from typing import List

import pytest

from langkit.core.workflow import MetricFilterOptions, RunOptions, Workflow
from langkit.metrics.library import lib
from langkit.validators.library import lib as validator_lib


def test_just_prompt():
    wf = Workflow(metrics=[lib.presets.recommended()])
    result = wf.run({"prompt": "hi"})
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
        "id",
    ]


def test_metric_filter_prompt():
    wf = Workflow(metrics=[lib.presets.recommended(), lib.response.similarity.prompt()])
    options = RunOptions(metric_filter=MetricFilterOptions(by_required_inputs=[["prompt"]]))
    result = wf.run({"prompt": "hi", "response": "hello"}, options)
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
        "id",
    ]


def test_metric_filter_response():
    wf = Workflow(metrics=[lib.presets.recommended(), lib.response.similarity.prompt()])
    options = RunOptions(metric_filter=MetricFilterOptions(by_required_inputs=[["response"]]))
    result = wf.run({"prompt": "hi", "response": "hello"}, options)
    metrics = result.metrics

    metric_names: List[str] = metrics.columns.tolist()  # pyright: ignore[reportUnknownMemberType]

    assert metric_names == [
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


def test_metric_filter_prompt_or_response():
    wf = Workflow(metrics=[lib.presets.recommended(), lib.response.similarity.prompt()])
    options = RunOptions(metric_filter=MetricFilterOptions(by_required_inputs=[["prompt"], ["response"]]))
    result = wf.run({"prompt": "hi", "response": "hello"}, options)
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

    # swap the order of response/prompt filter, should be the same output
    options_swapped = RunOptions(metric_filter=MetricFilterOptions(by_required_inputs=[["response"], ["prompt"]]))
    result = wf.run({"prompt": "hi", "response": "hello"}, options_swapped)
    assert metric_names == result.metrics.columns.tolist()  # pyright: ignore[reportUnknownMemberType]


def test_metric_filter_both_prompt_and_response():
    wf = Workflow(metrics=[lib.presets.recommended(), lib.response.similarity.prompt()])
    options = RunOptions(metric_filter=MetricFilterOptions(by_required_inputs=[["prompt", "response"]]))
    result = wf.run({"prompt": "hi", "response": "hello"}, options)
    metrics = result.metrics

    metric_names: List[str] = metrics.columns.tolist()  # pyright: ignore[reportUnknownMemberType]

    assert metric_names == [
        "response.similarity.prompt",
        "id",
    ]


def test_metric_filter_no_metrics_left():
    wf = Workflow(metrics=[lib.presets.recommended(), lib.response.similarity.prompt()])
    options = RunOptions(metric_filter=MetricFilterOptions(by_required_inputs=[["doesnt exist"]]))

    with pytest.raises(ValueError):
        wf.run({"prompt": "hi", "response": "hello"}, options)


def test_metric_filter_include_everything():
    wf = Workflow(metrics=[lib.presets.recommended(), lib.response.similarity.prompt()])
    options = RunOptions(metric_filter=MetricFilterOptions(by_required_inputs=[["prompt", "response"], ["prompt"], ["response"]]))
    result = wf.run({"prompt": "hi", "response": "hello"}, options)
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
        "response.similarity.prompt",
        "id",
    ]

    # Should be the same as not filtering at all
    result_all = wf.run({"prompt": "hi", "response": "hello"})
    assert metric_names == result_all.metrics.columns.tolist()  # pyright: ignore[reportUnknownMemberType]

    # And order doesn't matter
    options_swapped = RunOptions(metric_filter=MetricFilterOptions(by_required_inputs=[["response", "prompt"], ["response"], ["prompt"]]))
    result_swapped = wf.run({"prompt": "hi", "response": "hello"}, options_swapped)
    assert metric_names == result_swapped.metrics.columns.tolist()  # pyright: ignore[reportUnknownMemberType]


def test_just_prompt_validation():
    rule = validator_lib.constraint(target_metric="response.stats.token_count", upper_threshold=10)
    wf = Workflow(metrics=[lib.presets.recommended()], validators=[rule])

    result = wf.run({"prompt": "hi"})
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
        "id",
    ]


def test_just_response():
    wf = Workflow(metrics=[lib.presets.recommended()])
    result = wf.run({"response": "I'm doing great!"})
    metrics = result.metrics

    metric_names: List[str] = metrics.columns.tolist()  # pyright: ignore[reportUnknownMemberType]

    assert metric_names == [
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
