from logging import getLogger
import os
import tempfile

import pandas as pd
import pytest
import whylogs as why
from whylogs.experimental.core.udf_schema import udf_schema
from langkit import LangKitConfig


TEST_LOGGER = getLogger(__name__)


@pytest.fixture
def ptt_df():
    df = pd.DataFrame(
        {
            "prompt": [
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


@pytest.mark.parametrize("user_defined_json", [False, True])
def test_count_patterns(ptt_df, user_defined_json):
    from langkit import count_regexes

    if user_defined_json:
        with tempfile.TemporaryDirectory() as temp_dir:
            json_filename = "user.json"
            json_path = os.path.join(temp_dir, json_filename)
            with open(json_path, "w") as file:
                file.write(user_json)
            count_regexes.init(pattern_file_path=json_path, config=LangKitConfig())
    else:
        count_regexes.init(config=LangKitConfig())

    result = why.log(ptt_df, schema=udf_schema())
    view = result.view()

    if not user_defined_json:
        group_names = {
            "credit card number",
            "email address",
            "SSN",
            "phone number",
            "mailing address",
        }
    else:
        group_names = {"custom_group"}

    for group in group_names:
        assert (
            "distribution"
            in view.get_column(f"prompt.{group}_count").get_metric_names()
        )
