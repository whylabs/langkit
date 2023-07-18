from logging import getLogger
import os
import tempfile

import pandas as pd
import pytest
import whylogs as why
from whylogs.experimental.core.metrics.udf_metric import udf_metric_schema
from langkit.regexes import pattern_loader


TEST_LOGGER = getLogger(__name__)


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


@pytest.fixture
def target_pattern_tests():
    test_groups = {}
    test_groups["credit card number"] = [
        "credit card 4556205848969759",
        "credit card 3851-6256-0926-7271",
        "Visa Card Number: 4929 5423 7528 1067 \nExpiration Date: 03/24 \nCVV: 348",
        "622202049892743 - this is a credit card number",
    ]

    test_groups["email address"] = [
        "anemail@address.com",
        "my.email@whylabs.ai",
    ]

    test_groups["SSN"] = [
        "my ssn is 856-45-6789",
        "ssn - 702-02-9921",
        "ssn is 702 02 9921",
    ]

    test_groups["phone number"] = [
        "my phone is +1 309-404-7587",
        "Can you call me at (206) 555-1212?",
    ]

    test_groups["mailing address"] = [
        "address: 123 Main St.",
        "2255 140th Ave. NE",
        "535 Bellevue Sq",
        "15220 SE 37th St, its a nice place",
    ]

    return test_groups


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
            "credit card number",
            "email address",
            "SSN",
            "phone number",
            "mailing address",
        }
    else:
        group_names = {"custom_group"}

    assert set([x.value for x in fi_input_list]) == group_names


def test_individual_patterns_isolated(target_pattern_tests):
    from langkit import light_metrics

    test_schema = light_metrics.init()

    for target_pattern in target_pattern_tests:
        for test_prompt in target_pattern_tests[target_pattern]:
            result = why.log({"prompt": test_prompt}, schema=test_schema)
            if result.view().get_column("prompt").get_metric("udf") is None:
                TEST_LOGGER.warning(
                    f"No UDFs={test_schema.resolvers._resolvers} Failed to find pattern {target_pattern} in {test_prompt}"
                )
                TEST_LOGGER.info(result.view().get_column("prompt").to_summary_dict())
            frequentStringsComponent = (
                result.view()
                .get_column("prompt")
                .get_metric("udf")
                .submetrics.get("has_patterns")
                .get("frequent_items")
            )
            assert frequentStringsComponent is not None
            for frequent_item in frequentStringsComponent.strings:
                if target_pattern not in frequent_item.value:
                    TEST_LOGGER.warning(
                        f"Failed to find pattern {target_pattern} in {test_prompt}"
                        f"registered patterns are: {pattern_loader.get_regex_groups()}"
                    )
                    TEST_LOGGER.info(
                        result.view().get_column("prompt").to_summary_dict()
                    )
                assert target_pattern in frequent_item.value
