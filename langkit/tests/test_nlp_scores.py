import whylogs as why
import pytest


@pytest.mark.load
def test_bleu_score():
    from langkit import nlp_scores  # noqa
    from whylogs.experimental.core.udf_schema import udf_schema

    nlp_scores.init(
        scores=["bleu"], corpus="The quick brown fox jumps over the lazy dog"
    )
    text_schema = udf_schema()
    profile = why.log(
        {"response": "The quick dog jumps over the lazy brown fox"}, schema=text_schema
    ).profile()
    max_score = (
        profile.view()
        .get_column("response")
        .get_metrics()[-1]
        .to_summary_dict()["bleu_score:distribution/max"]
    )
    assert round(max_score, 2) == 0.42
