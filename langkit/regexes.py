from logging import getLogger

from langkit.pattern_loader import PatternLoader
from whylogs.experimental.core.udf_schema import register_dataset_udf
from . import LangKitConfig
from whylogs.core.metrics.metrics import FrequentItemsMetric
from whylogs.core.resolvers import MetricSpec
from whylogs.core.stubs import pd
from typing import Optional

diagnostic_logger = getLogger(__name__)


pattern_loader = PatternLoader()


def has_patterns(text):
    regex_groups = pattern_loader.get_regex_groups()
    result = []
    if regex_groups:
        index = (
            text.columns[0] if isinstance(text, pd.DataFrame) else list(text.keys())[0]
        )
        for input in text[index]:
            matched = None
            for group in regex_groups:
                for expression in group["expressions"]:
                    if expression.search(input):
                        matched = matched or group["name"]
                        break
                if matched is not None:
                    break

            result.append(matched)

    return result


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

    if pattern_loader.get_regex_groups() is not None:
        for column in [lang_config.prompt_column, lang_config.response_column]:
            register_dataset_udf(
                [column],
                udf_name=f"{column}.has_patterns",
                metrics=[MetricSpec(FrequentItemsMetric)],
            )(has_patterns)
