import whylogs as why
import pytest


@pytest.mark.load
def test_toxicity():
    from langkit import toxicity  # noqa
    from whylogs.experimental.core.udf_schema import udf_schema

    text_schema = udf_schema()
    profile = why.log(
        {"input": "I like you. I love you."}, schema=text_schema
    ).profile()
    mean_score = (
        profile.view()
        .get_column("input")
        .get_metrics()[-1]
        .to_summary_dict()["toxicity:distribution/mean"]
    )
    assert mean_score < 0.1


@pytest.mark.load
def test_toxicity_long_response(long_response):
    from langkit import toxicity  # noqa
    from whylogs.experimental.core.udf_schema import udf_schema

    text_schema = udf_schema()
    profile = why.log(long_response, schema=text_schema).profile()
    assert profile
