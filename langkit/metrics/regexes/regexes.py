import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from langkit.metrics.metric import Metric, MetricCreator, MetricResult, UdfInput
from langkit.metrics.regexes.regex_loader import CompiledPatternGroups, load_patterns_file

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


def __get_regexes_frequent_items_module(column_name: str, patterns: CompiledPatternGroups) -> Metric:
    def udf(text: pd.DataFrame) -> MetricResult:
        metric = [__has_any_patterns(patterns, it) for it in UdfInput(text).iter_column_rows(column_name)]
        return MetricResult(metric)

    return Metric(
        name=f"{column_name}.has_patterns",
        input_name=column_name,
        evaluate=udf,
    )


prompt_default_regexes_module = partial(__get_regexes_frequent_items_module, "prompt", __default_patterns)
response_default_regexes_module = partial(__get_regexes_frequent_items_module, "response", __default_patterns)
prompt_response_default_regexes_module = [prompt_default_regexes_module, response_default_regexes_module]


def get_custom_regex_frequent_items_for_column_module(
    column_name: str, file_or_patterns: Union[str, CompiledPatternGroups]
) -> MetricCreator:
    """
    This module getter is the same as get_custom_regex_frequent_items_modules but it lets you specify a column name,
    instead of assuming prompt/response.
    """
    if isinstance(file_or_patterns, str):
        patterns = load_patterns_file(file_or_patterns)
    else:
        patterns = file_or_patterns

    return partial(__get_regexes_frequent_items_module, column_name, patterns)


@dataclass(frozen=True)
class CustomRegexFreqItemsModules:
    prompt_custom_regexes_frequent_items_module: MetricCreator
    response_custom_regexes_frequent_items_module: MetricCreator
    prompt_response_custom_regexes_frequent_items_module: MetricCreator


def get_custom_regex_frequent_items_modules(file_or_patterns: Union[str, CompiledPatternGroups]) -> CustomRegexFreqItemsModules:
    """
    Get modules for running regex based tests on prompts and responses.
    The modules generated from this function will create "frequent items" metrics, which means that instead
    of seeing "prompt.email", "response.mailing_address", etc, you'll see "prompt.has_patterns" and "response.has_patterns".
    Those metrics will have a frequent item of "SSN" or "MAILING_ADDRESS" rather than be split out across n patterns.
    This is personal preference.
    """
    if isinstance(file_or_patterns, str):
        patterns = load_patterns_file(file_or_patterns)
    else:
        patterns = file_or_patterns

    prompt_custom_regexes_frequent_items_module = partial(__get_regexes_frequent_items_module, "prompt", patterns)
    response_custom_regexes_frequent_items_module = partial(__get_regexes_frequent_items_module, "response", patterns)

    return CustomRegexFreqItemsModules(
        prompt_custom_regexes_frequent_items_module=prompt_custom_regexes_frequent_items_module,
        response_custom_regexes_frequent_items_module=response_custom_regexes_frequent_items_module,
        prompt_response_custom_regexes_frequent_items_module=[
            prompt_custom_regexes_frequent_items_module,
            response_custom_regexes_frequent_items_module,
        ],
    )


def __single_regex_module(column_name: str, patterns: CompiledPatternGroups, pattern_name: str) -> Metric:
    def udf(text: Union[pd.DataFrame, Dict[str, List[Any]]]) -> MetricResult:
        metrics = [__has_pattern(patterns, pattern_name, it) for it in UdfInput(text).iter_column_rows(column_name)]
        return MetricResult(metrics)

    return Metric(
        name=f"{column_name}.{__sanitize_name_for_metric(pattern_name)}",
        input_name=column_name,
        evaluate=udf,
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


def get_custom_regex_for_column_module(column_name: str, file_or_patterns: Union[str, CompiledPatternGroups]) -> MetricCreator:
    """
    Using this to create a custom regex module will result in the generated metrics
    appearing as `prompt.<pattern_name>` and `response.<pattern_name>`, or whatever you supply
    for the column_name parameter.

    This function lets you specify any column name you want, rather than just prompt/response.
    """
    if isinstance(file_or_patterns, str):
        patterns = load_patterns_file(file_or_patterns)
    else:
        patterns = file_or_patterns

    return lambda: [__single_regex_module(column_name, patterns, pattern["name"]) for pattern in patterns["patterns"]]


@dataclass(frozen=True)
class CustomRegexModules:
    prompt_custom_regex_module: MetricCreator
    response_custom_regex_module: MetricCreator
    prompt_response_custom_regex_module: MetricCreator


# this "get_" pattern was introduced to clearly distinguish between the modules you can just use directly
# and the ones that require some input before you can use them. Otherwise, you had no idea based on the names alone
# Now, if it has "module" as a suffix, you can definitely put it into a schema builder as is.
def get_custom_regex_modules(file_path_or_pattern: Union[str, CompiledPatternGroups]) -> CustomRegexModules:
    """ """
    prompt_custom_regex_module = get_custom_regex_for_column_module("prompt", file_path_or_pattern)
    response_custom_regex_module = get_custom_regex_for_column_module("response", file_path_or_pattern)

    return CustomRegexModules(
        prompt_custom_regex_module=prompt_custom_regex_module,
        response_custom_regex_module=response_custom_regex_module,
        prompt_response_custom_regex_module=[prompt_custom_regex_module, response_custom_regex_module],
    )


# TODO can have a metric that returns multiple features, one being the substitution
def __pattern_substitution_match(patterns: CompiledPatternGroups, group_name: str, input: str) -> Optional[str]:
    pattern = next((it for it in patterns["patterns"] if it["name"] == group_name), None)

    if pattern is None:
        raise ValueError(f"Pattern group {group_name} not found in {patterns}")

    expressions = pattern["expressions"]
    substitutions = pattern["substitutions"]

    if not substitutions:
        # Validate for this at metric creation time as well
        raise ValueError(f"Pattern group {group_name} does not have substitutions")

    sub_str = input
    for expr, sub in zip(expressions, substitutions):
        sub_str = expr.sub(sub, sub_str)

    return sub_str if sub_str != input else None


def __single_substitution_metric(column_name: str, patterns: CompiledPatternGroups, pattern_name: str) -> Metric:
    def udf(text: pd.DataFrame) -> MetricResult:
        metrics = [__pattern_substitution_match(patterns, pattern_name, it) for it in UdfInput(text).iter_column_rows(column_name)]
        return MetricResult(metrics)

    return Metric(
        name=f"{column_name}.substitutions.{__sanitize_name_for_metric(pattern_name)}",
        input_name=column_name,
        evaluate=udf,
    )


def get_custom_substitutions(column_name: str, file_or_patterns: Union[str, CompiledPatternGroups]) -> MetricCreator:
    """
    Using this to create a custom regex module will result in the generated metrics
    appearing as `prompt.<pattern_name>` and `response.<pattern_name>`, or whatever you supply
    for the column_name parameter.

    This function lets you specify any column name you want, rather than just prompt/response.
    """
    if isinstance(file_or_patterns, str):
        patterns = load_patterns_file(file_or_patterns)
    else:
        patterns = file_or_patterns

    return lambda: [__single_substitution_metric(column_name, patterns, pattern["name"]) for pattern in patterns["patterns"]]
