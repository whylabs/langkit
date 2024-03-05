from logging import getLogger
import whylogs as why
import pytest


TEST_LOGGER = getLogger(__name__)


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


@pytest.mark.load
def test_empty_toxicity():
    from langkit import toxicity  # noqa
    from whylogs.experimental.core.udf_schema import udf_schema

    text_schema = udf_schema()
    test_prompt = "hi."
    test_response = ""
    test_message = {"prompt": test_prompt, "response": test_response}
    profile = why.log(test_message, schema=text_schema).profile()
    prompt_score = (
        profile.view()
        .get_column("prompt.toxicity")
        .get_metric("distribution")
        .to_summary_dict()["mean"]
    )
    response_score = (
        profile.view()
        .get_column("response.toxicity")
        .get_metric("distribution")
        .to_summary_dict()["mean"]
    )

    TEST_LOGGER.info(f"running toxicity metrics on {test_message}")
    TEST_LOGGER.info(
        f"prompt score is: {prompt_score} and response score is: {response_score}"
    )
    assert prompt_score < 0.1
    assert response_score < 0.1


@pytest.mark.load
def test_toxicity_detoxify():
    from langkit import toxicity, extract  # noqa

    toxicity.init(model_path="detoxify/unbiased")
    result = extract({"prompt": "I like you. I love you."})
    assert result["prompt.toxicity"] < 0.1

    toxicity.init(model_path="detoxify/original")
    result = extract({"prompt": "I like you. I love you."})
    assert result["prompt.toxicity"] < 0.1

    toxicity.init(model_path="detoxify/multilingual")
    result = extract({"prompt": "Eu gosto de você. Eu te amo."})
    assert result["prompt.toxicity"] < 0.1


@pytest.mark.load
def test_toxicity_unknown_model():
    from langkit import toxicity  # noqa

    with pytest.raises(ValueError):
        toxicity.init(model_path="unknown")
