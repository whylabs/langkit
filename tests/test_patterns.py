import pandas as pd
import whylogs as why
from whylogs.core.resolvers import STANDARD_RESOLVER
from whylogs.core.schema import DeclarativeSchema
from whylogs.experimental.core.metrics.udf_metric import generate_udf_schema
from langkit import regexes
import pytest


@pytest.fixture
def pii_df():
    return pd.DataFrame(
        {
            "input": ["no_info", "my ssn is 222-11-4444", "22-11-33"],
            "output": ["a", "b", "c"],
        }
    )


# log dataframe
def test_pii(pii_df):
    schema = DeclarativeSchema(STANDARD_RESOLVER + generate_udf_schema())
    result = why.log(pii_df, schema=schema)

    assert (
        result.view()
        .to_pandas()["udf/has_patterns:frequent_items/frequent_strings"]["input"][0]
        .value
        == "SSN"
    )
    assert (
        result.view()
        .to_pandas()["udf/has_patterns:frequent_items/frequent_strings"]["input"][1]
        .value
        == "Email"
    )
    assert (
        result.view()
        .to_pandas()["udf/has_patterns:frequent_items/frequent_strings"]["input"][2]
        .value
        == ""
    )
    assert (
        result.view()
        .to_pandas()["udf/has_patterns:frequent_items/frequent_strings"]["output"][0]
        .value
        == ""
    )
