import pytest
import whylogs as why
from whylogs.core.metrics import MetricConfig
from whylogs.experimental.core.udf_schema import generate_udf_dataset_schema


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
def test_similarity(interactions):
    # default input col is "prompt" and output col is "response".
    from langkit import input_output as lkio

    schema = generate_udf_dataset_schema(default_config=MetricConfig(fi_disabled=True))
    for i, interaction in enumerate(interactions):
        result = why.log(interaction, schema=schema)
        column_name = f"similarity_{lkio._transformer_name.split('/')[-1]}"
        if {"prompt", "response"}.issubset(set(interaction.keys())):
            similarity_median = (
                result.view()
                .get_column(column_name)
                .get_metric("distribution")
                .to_summary_dict()["median"]
            )
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
