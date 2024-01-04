import os
from functools import partial
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from langkit.module.module import Module, ModuleFn, UdfInput, UdfSchemaArgs
from langkit.module.regexes.regex_loader import CompiledPatternGroups, load_patterns_file
from whylogs.core.resolvers import NO_FI_RESOLVER, MetricSpec, ResolverSpec, StandardMetric
from whylogs.experimental.core.udf_schema import UdfSpec

__current_module_path = os.path.dirname(__file__)
__default_pattern_file = os.path.join(__current_module_path, "pattern_groups.json")
__default_patterns: CompiledPatternGroups = load_patterns_file(__default_pattern_file)


def __sanitize_name_for_metric(pattern_name: str) -> str:
    return pattern_name.replace(" ", "_").lower()


def __sanitize_name_for_frequent_item(pattern_name: str) -> str:
    return pattern_name.replace(" ", "_").upper()


def has_any_patterns(patterns: CompiledPatternGroups, input: str) -> Optional[str]:
    for pattern in patterns["patterns"]:
        if any(expr.search(input) for expr in pattern["expressions"]):
            return __sanitize_name_for_frequent_item(pattern["name"])

    return None


def has_pattern(patterns: CompiledPatternGroups, group_name: str, input: str) -> int:
    pattern = next((it for it in patterns["patterns"] if it["name"] == group_name), None)

    if pattern is None:
        raise ValueError(f"Pattern group {group_name} not found in {patterns}")

    if any(expr.search(input) for expr in pattern["expressions"]):
        return 1

    return 0


def __regexes_frequent_items_module(column_name: str, patterns: CompiledPatternGroups) -> UdfSchemaArgs:
    def _udf(column_name: str, text: Union[pd.DataFrame, Dict[str, List[Any]]]) -> Any:
        return [has_any_patterns(patterns, it) for it in UdfInput(text).iter_column(column_name)]

    udf = partial(_udf, column_name)

    udf_column_name = f"{column_name}.has_patterns"

    spec = UdfSpec(
        column_names=[column_name],
        udfs={udf_column_name: udf},
    )

    return UdfSchemaArgs(
        types={column_name: str},
        resolvers=[
            ResolverSpec(
                column_name=udf_column_name,
                metrics=[MetricSpec(StandardMetric.frequent_items.value)],
            )
        ],
        udf_specs=[spec],
    )


prompt_default_regexes = partial(__regexes_frequent_items_module, "prompt", __default_patterns)
response_default_regexes = partial(__regexes_frequent_items_module, "response", __default_patterns)
prompt_response_default_regexes = [prompt_default_regexes, response_default_regexes]


def __custom_regexes_frequent_items_module(column_name: str, file_or_patterns: Union[str, CompiledPatternGroups]) -> ModuleFn:
    if isinstance(file_or_patterns, str):
        patterns = load_patterns_file(file_or_patterns)
    else:
        patterns = file_or_patterns

    return lambda: __regexes_frequent_items_module(column_name, patterns)


prompt_custom_regexes_frequent_items_module = partial(__custom_regexes_frequent_items_module, "prompt")
response_custom_regexes_frequent_items_module = partial(__custom_regexes_frequent_items_module, "response")


def prompt_response_custom_regex_frequent_items_module(file_path: str) -> Module:
    return [
        prompt_custom_regexes_frequent_items_module(file_path),
        response_custom_regexes_frequent_items_module(file_path),
    ]


def __single_regex_module(column_name: str, patterns: CompiledPatternGroups, pattern_name: str) -> UdfSchemaArgs:
    def _udf(column_name: str, text: Union[pd.DataFrame, Dict[str, List[Any]]]) -> Any:
        return [has_pattern(patterns, pattern_name, it) for it in UdfInput(text).iter_column(column_name)]

    udf = partial(_udf, column_name)

    spec = UdfSpec(
        column_names=[column_name],
        udfs={f"{column_name}.{__sanitize_name_for_metric(pattern_name)}": udf},
    )

    return UdfSchemaArgs(
        types={column_name: str},
        resolvers=NO_FI_RESOLVER,
        udf_specs=[spec],
    )


prompt_ssn_regex = partial(__single_regex_module, "prompt", __default_patterns, "SSN")
prompt_credit_card_number_regex = partial(__single_regex_module, "prompt", __default_patterns, "credit card number")
prompt_phone_number_regex = partial(__single_regex_module, "prompt", __default_patterns, "phone number")
prompt_mailing_address_regex = partial(__single_regex_module, "prompt", __default_patterns, "mailing address")
prompt_email_address_regex = partial(__single_regex_module, "prompt", __default_patterns, "email address")

response_ssn_regex = partial(__single_regex_module, "response", __default_patterns, "SSN")
response_credit_card_number_regex = partial(__single_regex_module, "response", __default_patterns, "credit card number")
response_phone_number_regex = partial(__single_regex_module, "response", __default_patterns, "phone number")
response_mailing_address_regex = partial(__single_regex_module, "response", __default_patterns, "mailing address")
response_email_address_regex = partial(__single_regex_module, "response", __default_patterns, "email address")

prompt_response_ssn_regex = [prompt_ssn_regex, response_ssn_regex]
prompt_response_credit_card_number_regex = [prompt_credit_card_number_regex, response_credit_card_number_regex]
prompt_response_phone_number_regex = [prompt_phone_number_regex, response_phone_number_regex]
prompt_response_mailing_address_regex = [prompt_mailing_address_regex, response_mailing_address_regex]
prompt_response_email_address_regex = [prompt_email_address_regex, response_email_address_regex]


def custom_regex_module(column_name: str, file_or_patterns: Union[str, CompiledPatternGroups]):
    """
    Create a custom regex module directly with a pattern group.
    Using this to create a custom regex module will result in the generated metrics
    appearing as `prompt.<pattern_name>` and `response.<pattern_name>`, or whatever you supply
    for the column_name parameter.
    """
    if isinstance(file_or_patterns, str):
        patterns = load_patterns_file(file_or_patterns)
    else:
        patterns = file_or_patterns

    return lambda: [__single_regex_module(column_name, patterns, pattern["name"]) for pattern in patterns["patterns"]]


prompt_custom_regex_module = partial(custom_regex_module, "prompt")
response_custom_regex_module = partial(custom_regex_module, "response")
prompt_response_custom_regex_module = [prompt_custom_regex_module, response_custom_regex_module]
