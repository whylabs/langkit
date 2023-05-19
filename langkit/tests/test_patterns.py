import os
import tempfile

import pandas as pd
import pytest
import whylogs as why
from whylogs.experimental.core.metrics.udf_metric import udf_metric_schema


@pytest.fixture
def ptt_df():
    df = pd.DataFrame(
        {
            "input": [
                "address: 123 Main St.",
                "2255 140th Ave. NE",
                "535 Bellevue Sq",
                "15220 SE 37th St, its a nice place",
                "anemail@address.com",
                "my phone is +1 309-404-7587",
                "credit card 4556205848969759",
                "credit card 3851-6256-0926-7271",
                "Visa Card Number: 4929 5423 7528 1067 \nExpiration Date: 03/24 \nCVV: 348",
                "622202049892743 - this is a credit card number",
                "my ssn is 856-45-6789",
                "ssn - 702-02-9921",
                "ssn is 702 02 9921",
                "702029921 (SSN)",
                "no patterns here.",
            ]
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
    from langkit import regexes

    if user_defined_json:
        with tempfile.TemporaryDirectory() as temp_dir:
            json_filename = "user.json"
            json_path = os.path.join(temp_dir, json_filename)
            with open(json_path, "w") as file:
                file.write(user_json)
            regexes.init(pattern_file_path=os.path.join(temp_dir, json_filename))

    result = why.log(ptt_df, schema=udf_metric_schema())
    fi_input_list = result.view().to_pandas()[
        "udf/has_patterns:frequent_items/frequent_strings"
    ]["input"]
    if not user_defined_json:
        group_names = {
            "",
            "credit card number",
            "email address",
            "SSN",
            "phone number",
            "mailing address",
        }
    else:
        group_names = {"custom_group", ""}

    assert set([x.value for x in fi_input_list]) == group_names
