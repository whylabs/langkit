import pytest
import whylogs as why
from whylogs.core.metrics import MetricConfig
from whylogs.experimental.core.udf_schema import udf_schema
from typing import List


@pytest.fixture
def interactions():
    interactions_list = [
        {"prompt": "P"},
        {"response": "I"},
        {"prompt": "What?", "response": "blue"},
        {"prompt": ["W"], "response": ["b"]},
        {"prompt": 3, "response": ["b"]},
        {"prompt": "", "response": "b"},
        {"prompt": "c", "response": ""},
        {"prompt": "", "response": ""},
        {"prompt": None, "response": ""},
        {"prompt": "N", "response": None},
        {"prompt": "I", "response": 4},
    ]
    return interactions_list


def texty(d):
    try:
        for v in d.values():
            if not (isinstance(v, str) or all([isinstance(x, str) for x in v])):
                return False
    except Exception:
        return False

    return True


@pytest.mark.load
def test_init_call():
    from langkit import input_output

    with pytest.raises(ValueError):
        input_output.init(
            transformer_name="sentence-transformers/all-MiniLM-L6-v2",
            custom_encoder=lambda x: [[0.2, 0.2] for _ in x],
        )


@pytest.mark.load
def test_custom_encoder():
    from langkit import input_output

    def embed(texts: List[str]):
        return [[0.2, 0.2] for _ in texts]

    interaction = {
        "prompt": "What?",
        "response": "blue",
    }
    input_output.init(custom_encoder=embed)
    schema = udf_schema()
    result = why.log(interaction, schema=schema)
    dist_metric = (
        result.view()
        .get_column("response.relevance_to_prompt")
        .get_metric("distribution")
    )
    assert dist_metric.to_summary_dict()["mean"] == pytest.approx(1.0)


@pytest.mark.load
def test_similarity(interactions):
    # default input col is "prompt" and output col is "response".
    from langkit import input_output as lkio  # noqa
    from whylogs.experimental.core.udf_schema import logger as uslog

    iolog = lkio.diagnostic_logger
    old_uslevel = uslog.getEffectiveLevel()
    old_iolevel = iolog.getEffectiveLevel()
    uslog.setLevel("CRITICAL")  # intentionally feeding bad input, so be quiet about it
    iolog.setLevel("CRITICAL")

    lkio.init()
    schema = udf_schema(default_config=MetricConfig(fi_disabled=True))
    for i, interaction in enumerate(interactions):
        result = why.log(interaction, schema=schema)
        column_name = "response.relevance_to_prompt"
        if {"prompt", "response"}.issubset(set(interaction.keys())):
            similarity_median = (
                result.view()
                .get_column(column_name)
                .get_metric("distribution")
                .to_summary_dict()["median"]
            )
            if similarity_median is None:
                print(interaction)
                print(result.view().get_column(column_name).to_summary_dict())
            assert (similarity_median is None and not texty(interaction)) or (
                texty(interaction) and (-1.1 <= similarity_median <= 1.1)
            )
            assert (
                "frequent_items/frequent_strings"
                not in result.view().get_column("prompt").to_summary_dict()
            )
            assert (
                "frequent_items/frequent_strings"
                not in result.view().get_column("response").to_summary_dict()
            )
        else:
            assert column_name not in result.view().get_columns()

    iolog.setLevel(old_iolevel)
    uslog.setLevel(old_uslevel)
