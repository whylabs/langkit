import os
from dataclasses import dataclass
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


def __has_any_patterns(patterns: CompiledPatternGroups, input: str) -> Optional[str]:
    for pattern in patterns["patterns"]:
        if any(expr.search(input) for expr in pattern["expressions"]):
            return __sanitize_name_for_frequent_item(pattern["name"])

    return None


def __has_pattern(patterns: CompiledPatternGroups, group_name: str, input: str) -> int:
    pattern = next((it for it in patterns["patterns"] if it["name"] == group_name), None)

    if pattern is None:
        raise ValueError(f"Pattern group {group_name} not found in {patterns}")

    if any(expr.search(input) for expr in pattern["expressions"]):
        return 1

    return 0


def __regexes_frequent_items_module(column_name: str, patterns: CompiledPatternGroups) -> UdfSchemaArgs:
    def _udf(column_name: str, text: Union[pd.DataFrame, Dict[str, List[Any]]]) -> Any:
        return [__has_any_patterns(patterns, it) for it in UdfInput(text).iter_column(column_name)]

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


prompt_default_regexes_module = partial(__regexes_frequent_items_module, "prompt", __default_patterns)
response_default_regexes_module = partial(__regexes_frequent_items_module, "response", __default_patterns)
prompt_response_default_regexes_module = [prompt_default_regexes_module, response_default_regexes_module]


def __custom_regexes_frequent_items_module(column_name: str, file_or_patterns: Union[str, CompiledPatternGroups]) -> ModuleFn:
    if isinstance(file_or_patterns, str):
        patterns = load_patterns_file(file_or_patterns)
    else:
        patterns = file_or_patterns

    return lambda: __regexes_frequent_items_module(column_name, patterns)


@dataclass(frozen=True)
class CustomRegexFreqItemsModules:
    prompt_custom_regexes_frequent_items_module: Module
    response_custom_regexes_frequent_items_module: Module
    prompt_response_custom_regexes_frequent_items_module: Module


def get_custom_regex_frequent_items_modules(file_path: str) -> CustomRegexFreqItemsModules:
    prompt_custom_regexes_frequent_items_module = __custom_regexes_frequent_items_module("prompt", file_path)
    response_custom_regexes_frequent_items_module = __custom_regexes_frequent_items_module("response", file_path)

    return CustomRegexFreqItemsModules(
        prompt_custom_regexes_frequent_items_module=prompt_custom_regexes_frequent_items_module,
        response_custom_regexes_frequent_items_module=response_custom_regexes_frequent_items_module,
        prompt_response_custom_regexes_frequent_items_module=[
            prompt_custom_regexes_frequent_items_module,
            response_custom_regexes_frequent_items_module,
        ],
    )


def __single_regex_module(column_name: str, patterns: CompiledPatternGroups, pattern_name: str) -> UdfSchemaArgs:
    def _udf(column_name: str, text: Union[pd.DataFrame, Dict[str, List[Any]]]) -> Any:
        return [__has_pattern(patterns, pattern_name, it) for it in UdfInput(text).iter_column(column_name)]

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


prompt_ssn_regex_module = partial(__single_regex_module, "prompt", __default_patterns, "SSN")
prompt_credit_card_number_regex_module = partial(__single_regex_module, "prompt", __default_patterns, "credit card number")
prompt_phone_number_regex_module = partial(__single_regex_module, "prompt", __default_patterns, "phone number")
prompt_mailing_address_regex_module = partial(__single_regex_module, "prompt", __default_patterns, "mailing address")
prompt_email_address_regex_module = partial(__single_regex_module, "prompt", __default_patterns, "email address")

response_ssn_regex_module = partial(__single_regex_module, "response", __default_patterns, "SSN")
response_credit_card_number_regex_module = partial(__single_regex_module, "response", __default_patterns, "credit card number")
response_phone_number_regex_module = partial(__single_regex_module, "response", __default_patterns, "phone number")
response_mailing_address_regex_module = partial(__single_regex_module, "response", __default_patterns, "mailing address")
response_email_address_regex_module = partial(__single_regex_module, "response", __default_patterns, "email address")

prompt_response_ssn_regex_module = [prompt_ssn_regex_module, response_ssn_regex_module]
prompt_response_credit_card_number_regex_module = [prompt_credit_card_number_regex_module, response_credit_card_number_regex_module]
prompt_response_phone_number_regex_module = [prompt_phone_number_regex_module, response_phone_number_regex_module]
prompt_response_mailing_address_regex_module = [prompt_mailing_address_regex_module, response_mailing_address_regex_module]
prompt_response_email_address_regex_module = [prompt_email_address_regex_module, response_email_address_regex_module]


def custom_regex_module(column_name: str, file_or_patterns: Union[str, CompiledPatternGroups]) -> Module:
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


@dataclass(frozen=True)
class CustomRegexModules:
    prompt_custom_regex_module: Module
    response_custom_regex_module: Module
    prompt_response_custom_regex_module: Module


# this "get_" pattern was introduced to clearly distinguish between the modules you can just use directly
# and the ones that require some input before you can use them. Otherwise, you had no idea based on the names alone
# Now, if it has "module" as a suffix, you can definitely put it into a schema builder as is.
def get_custom_regex_modules(file_path_or_pattern: Union[str, CompiledPatternGroups]) -> CustomRegexModules:
    prompt_custom_regex_module = custom_regex_module("prompt", file_path_or_pattern)
    response_custom_regex_module = custom_regex_module("response", file_path_or_pattern)

    return CustomRegexModules(
        prompt_custom_regex_module=prompt_custom_regex_module,
        response_custom_regex_module=response_custom_regex_module,
        prompt_response_custom_regex_module=lambda: [prompt_custom_regex_module, response_custom_regex_module],
    )
