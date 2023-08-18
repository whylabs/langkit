import json
import re
from logging import getLogger

from langkit.pattern_loader import PatternLoader
from whylogs.experimental.core.udf_schema import register_dataset_udf
from . import LangKitConfig
from whylogs.core.metrics.metrics import FrequentItemsMetric
from whylogs.core.resolvers import MetricSpec
from whylogs.core.stubs import pd
from typing import Dict, List, Optional, Set, Union

diagnostic_logger = getLogger(__name__)


pattern_loader = PatternLoader()


def count_patterns(group, text: str) -> int:
    count = 0
    for expression in group["expressions"]:
        if expression.search(text):
            count += 1

    return count


def wrapper(pattern_group, column):

    def wrappee(text: Union[pd.DataFrame, Dict[str, List]]) -> Union[pd.Series, List]:
        return [count_patterns(pattern_group, input) for input in text[column]]

    return wrappee


_registered: Set[str] = set()


def _unregister():
    # WARNING: UNSUPPORTED HEINOUS EVIL
    from whylogs.experimental.core.udf_schema import _multicolumn_udfs
    global _multicolumn_udfs, _registered
    _multicolumn_udfs[""] = [u for u in _multicolumn_udfs[""] if list(u.udfs.keys())[0] in _registered]
    _registered = set()


def _register_udfs(lang_config: LangKitConfig):
    global _registered
    _unregister()
    regex_groups = pattern_loader.get_regex_groups()
    if regex_groups is not None:
        for column in [lang_config.prompt_column, lang_config.response_column]:
            for group in regex_groups:
                udf_name = f"{column}.{group['name']}_count"
                register_dataset_udf(
                    [column],
                    udf_name=udf_name,
                )(wrapper(group, column))
                _registered.add(udf_name)


def init(
    pattern_file_path: Optional[str] = None, lang_config: Optional[LangKitConfig] = None
):
    if lang_config is None:
        lang_config = LangKitConfig()
    if pattern_file_path:
        lang_config.pattern_file_path = pattern_file_path
        pattern_loader.set_config(lang_config)
        pattern_loader.update_patterns()
    else:
        pattern_loader.set_config(lang_config)
        pattern_loader.update_patterns()

    _register_udfs(lang_config)


init()
