from copy import deepcopy
from logging import getLogger

from langkit.pattern_loader import PatternLoader
from whylogs.experimental.core.udf_schema import register_dataset_udf
from langkit import LangKitConfig, lang_config, prompt_column, response_column
from whylogs.core.metrics.metrics import FrequentItemsMetric
from whylogs.core.resolvers import MetricSpec
from typing import Dict, List, Optional
from langkit.utils import _unregister_metric_udf

diagnostic_logger = getLogger(__name__)

pattern_loader = PatternLoader()


def has_patterns(text):
    regex_groups = pattern_loader.get_regex_groups()
    if regex_groups:
        matched = None
        for group in regex_groups:
            for expression in group["expressions"]:
                if expression.search(text):
                    matched = matched or group["name"]
                    break
            if matched is not None:
                break

        return matched


def _wrapper(column):
    def wrappee(text):
        return [has_patterns(input) for input in text[column]]

    return wrappee


_registered: List[str] = []


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
        for column in [prompt_column, response_column]:
            udf_name = f"{column}.{pattern_metric_name}"
            register_dataset_udf(
                [column],
                udf_name=udf_name,
                metrics=[MetricSpec(FrequentItemsMetric)],
            )(_wrapper(column))
            _registered.append(udf_name)


def init(
    pattern_file_path: Optional[str] = None, config: Optional[LangKitConfig] = None
):
    config = deepcopy(config or lang_config)
    if pattern_file_path:
        config.pattern_file_path = pattern_file_path

    global pattern_loader
    pattern_loader = PatternLoader(config)
    pattern_loader.update_patterns()

    _register_udfs(config)


init()
