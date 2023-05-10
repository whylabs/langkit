import pandas as pd
import whylogs as why
from whylogs.core.resolvers import STANDARD_RESOLVER
from whylogs.core.schema import DeclarativeSchema
from whylogs.experimental.core.metrics.udf_metric import generate_udf_schema
from langkit import regexes, LangKitConfig
import pytest
import tempfile
import os


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


user_json = """
[
    {
        "expressions": [
            "[A-Za-z0-9]+@[A-Za-z0-9]+.[A-Za-z]{2,}"
        ],
        "name": "custom_group"
    }
]
"""

# log dataframe
@pytest.mark.parametrize("user_defined_json", [False, True])
def test_ptt(ptt_df, user_defined_json):
    if user_defined_json:
        with tempfile.TemporaryDirectory() as temp_dir:
            json_filename = "user.json"
            json_path = os.path.join(temp_dir, json_filename)
            with open(json_path, "w") as file:
                file.write(user_json)
            regexes.set_config(
                LangKitConfig(pattern_file_path=os.path.join(temp_dir, json_filename))
            )

    schema = DeclarativeSchema(STANDARD_RESOLVER + generate_udf_schema())
    result = why.log(ptt_df, schema=schema)
    fi_input_list = result.view().to_pandas()[
        "udf/has_patterns:frequent_items/frequent_strings"
    ]["input"]
    fi_output_list = result.view().to_pandas()[
        "udf/has_patterns:frequent_items/frequent_strings"
    ]["output"]
    if not user_defined_json:
        group_names = {"phone number", "email address", "SSN", "mailing address", "credit card number"}
    else:
        group_names = {"custom_group", ""}

    assert set([x.value for x in fi_input_list]) == group_names
    assert set([x.value for x in fi_output_list]) == {""}
