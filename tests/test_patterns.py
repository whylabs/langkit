import pandas as pd
import whylogs as why
from whylogs.core.resolvers import STANDARD_RESOLVER
from whylogs.core.schema import DeclarativeSchema
from whylogs.experimental.core.metrics.udf_metric import generate_udf_schema
from langkit import regexes
import pytest


@pytest.fixture
def ptt_df():
    df = pd.DataFrame(
        {
            "input": [
                "address: 123 Main St Anytown, NY 12345",
                "anemail@address.com",
                "my phone is +1 309-404-7587",
                "credit card 4556205848969759",
                "my ssn is 856-45-6789",
            ],
            "output": ["a", "b", "c", "d", "e"],
        }
    )
    return df


# log dataframe
def test_ptt(ptt_df):
    schema = DeclarativeSchema(STANDARD_RESOLVER + generate_udf_schema())
    result = why.log(ptt_df, schema=schema)
    fi_input_list = result.view().to_pandas()[
        "udf/has_patterns:frequent_items/frequent_strings"
    ]["input"]
    fi_output_list = result.view().to_pandas()[
        "udf/has_patterns:frequent_items/frequent_strings"
    ]["output"]
    group_names = {"Telephone", "Email", "SSN", "US Address", "Payment Card"}
    assert set([x.value for x in fi_input_list]) == group_names
    assert set([x.value for x in fi_output_list]) == {""}
