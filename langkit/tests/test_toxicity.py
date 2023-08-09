import whylogs as why
import pytest


@pytest.mark.load
def test_toxicity():
    from langkit import toxicity  # noqa
    from whylogs.experimental.core.udf_schema import udf_schema

    text_schema = udf_schema()
    profile = why.log(
        {"prompt": "I like you. I love you."}, schema=text_schema
    ).profile()
    mean_score = (
        profile.view()
        .get_column("prompt.toxicity")
        .get_metric("distribution")
        .to_summary_dict()["mean"]
    )
    assert mean_score < 0.1


@pytest.mark.load
def test_toxicity_long_response(long_response):
    from langkit import toxicity  # noqa
    from whylogs.experimental.core.udf_schema import udf_schema

    text_schema = udf_schema()
    profile = why.log(long_response, schema=text_schema).profile()
    assert profile
