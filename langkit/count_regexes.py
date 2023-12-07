from collections import defaultdict
from copy import deepcopy
from logging import getLogger

from langkit.pattern_loader import PatternLoader
from whylogs.experimental.core.udf_schema import register_dataset_udf
from langkit import LangKitConfig, lang_config, prompt_column, response_column
from whylogs.core.stubs import pd
from typing import Dict, List, Optional, Set, Union
from langkit.whylogs.unreg import unregister_udfs  # replace with whylogs 1.3.12

diagnostic_logger = getLogger(__name__)


pattern_loader = PatternLoader()
response_pattern_loader = PatternLoader()

_initialized = False


def count_patterns(group, text: str) -> int:
    if not _initialized:
        init()
    count = 0
    for expression in group["expressions"]:
        if expression.search(text):
            count += 1

    return count


def wrapper(pattern_group, column):
    def wrappee(text: Union[pd.DataFrame, Dict[str, List]]) -> Union[pd.Series, List]:
        return [count_patterns(pattern_group, input) for input in text[column]]

    return wrappee


_registered: Dict[str, Set[str]] = defaultdict(
    set
)  # _registered[schema_name] -> set of registered UDF names


def _register_udfs(language: str, schema_name: str):
    global _registered
    unregister_udfs(_registered[schema_name], language, schema_name)
    _registered[schema_name] = set()

    regex_groups = pattern_loader.get_regex_groups()
    if regex_groups is not None:
        column = prompt_column
        for group in regex_groups:
            udf_name = f"{column}.{group['name']}_count"
            register_dataset_udf(
                [column],
                udf_name=udf_name,
                namespace=language,
                schema_name=schema_name,
            )(wrapper(group, column))
            _registered[schema_name].add(udf_name)

    regex_groups = response_pattern_loader.get_regex_groups()
    if regex_groups is not None:
        column = response_column
        for group in regex_groups:
            udf_name = f"{column}.{group['name']}_count"
            register_dataset_udf(
                [column],
                udf_name=udf_name,
                namespace=language,
                schema_name=schema_name,
            )(wrapper(group, column))
            _registered[schema_name].add(udf_name)


def init(
    language: Optional[str] = None,
    pattern_file_path: Optional[str] = None,
    config: Optional[LangKitConfig] = None,
    response_pattern_file_path: Optional[str] = None,
    schema_name: str = "",
):
    global _initialized
    _initialized = True
    language = language or ""
    config = deepcopy(config or lang_config)
    if pattern_file_path:
        config.pattern_file_path = pattern_file_path
    if response_pattern_file_path:
        config.response_pattern_file_path = response_pattern_file_path
    global pattern_loader, response_pattern_loader
    pattern_loader = PatternLoader(config.pattern_file_path)
    response_pattern_loader = PatternLoader(config.response_pattern_file_path)
    _register_udfs(language, schema_name)
