from copy import deepcopy
from logging import getLogger

from langkit.pattern_loader import PatternLoader
from whylogs.experimental.core.udf_schema import register_dataset_udf
from . import LangKitConfig, lang_config, prompt_column, response_column
from whylogs.core.metrics.metrics import FrequentItemsMetric
from whylogs.core.resolvers import MetricSpec
from typing import Dict, List, Optional

diagnostic_logger = getLogger(__name__)

pattern_loader = PatternLoader()
response_pattern_loader = PatternLoader()


def has_patterns(text, regex_groups):
    if regex_groups:
        for group in regex_groups:
            for expression in group["expressions"]:
                if expression.search(text):
                    return group["name"]
    return None


def _wrapper(column, groups):
    def wrappee(text):
        return [has_patterns(input, groups) for input in text[column]]

    return wrappee


_registered: List[str] = []


def _unregister_metric_udf(old_name: str, namespace: Optional[str] = ""):
    from whylogs.experimental.core.udf_schema import _multicolumn_udfs

    if _multicolumn_udfs is None or namespace not in _multicolumn_udfs:
        return

    _multicolumn_udfs[namespace] = [
        udf
        for udf in _multicolumn_udfs[namespace]
        if list(udf.udfs.keys())[0] != old_name
    ]


def _register_udfs(config: Optional[LangKitConfig] = None):
    from whylogs.experimental.core.udf_schema import _resolver_specs

    global _registered
    if _registered and config is None:
        return
    if config is None:
        config = lang_config
    default_metric_name = "has_patterns"
    pattern_metric_name = config.metric_name_map.get(
        default_metric_name, default_metric_name
    )

    for old in _registered:
        _unregister_metric_udf(old_name=old)
        if (
            _resolver_specs is not None
            and isinstance(_resolver_specs, Dict)
            and isinstance(_resolver_specs[""], List)
        ):
            _resolver_specs[""] = [
                spec for spec in _resolver_specs[""] if spec.column_name != old
            ]
    _registered = []

    if pattern_loader.get_regex_groups() is not None:
        column = prompt_column
        udf_name = f"{column}.{pattern_metric_name}"
        register_dataset_udf(
            [column],
            udf_name=udf_name,
            metrics=[MetricSpec(FrequentItemsMetric)],
        )(_wrapper(column, pattern_loader.get_regex_groups()))
        _registered.append(udf_name)

    if response_pattern_loader.get_regex_groups() is not None:
        column = response_column
        udf_name = f"{column}.{pattern_metric_name}"
        register_dataset_udf(
            [column],
            udf_name=udf_name,
            metrics=[MetricSpec(FrequentItemsMetric)],
        )(_wrapper(column, response_pattern_loader.get_regex_groups()))
        _registered.append(udf_name)


def init(
    language: Optional[str] = None,
    pattern_file_path: Optional[str] = None,
    config: Optional[LangKitConfig] = None,
    response_pattern_file_path: Optional[str] = None,
):
    config = deepcopy(config or lang_config)
    if pattern_file_path:
        config.pattern_file_path = pattern_file_path
    if response_pattern_file_path:
        config.response_pattern_file_path = response_pattern_file_path

    global pattern_loader, response_pattern_loader
    pattern_loader = PatternLoader(config.pattern_file_path)
    response_pattern_loader = PatternLoader(config.response_pattern_file_path)

    _register_udfs(config)
