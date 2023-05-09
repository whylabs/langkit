from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import (
    register_metric_udf,
)
import json
import re
import pkg_resources
from typing import Optional
from logging import getLogger
from . import LangKitConfig
from dataclasses import dataclass

logger = getLogger(__name__)

lang_config = LangKitConfig()

# @dataclass
# class PatternLoader:
#     pattern_file_path: str =


# pattern_loader = PatternLoader()
# If not stated otherwise, suggested patterns in the default file should reflect US patterns.
def load_patterns():
    json_path = lang_config.pattern_file_path
    try:
        skip = False
        with open(json_path, "r") as myfile:
            _REGEX_GROUPS = json.load(myfile)
        regex_groups = []
        for group in _REGEX_GROUPS:
            compiled_expressions = []
            for expression in group["expressions"]:
                compiled_expressions.append(re.compile(expression))
            regex_groups.append(
                {"name": group["name"], "expressions": compiled_expressions}
            )
    except FileNotFoundError:
        skip = True
        logger.warning(f"Could not find {json_path}")
    except json.decoder.JSONDecodeError:
        skip = True
        logger.warning(f"Could not parse {json_path}")
    if not skip:
        return regex_groups
    return None


regex_groups = load_patterns()


if regex_groups is not None:

    @register_metric_udf(col_type=String)
    def has_patterns(text: str) -> Optional[str]:
        global regex_groups
        patterns_info = ""
        for group in regex_groups:
            for expression in group["expressions"]:
                if expression.search(text):
                    patterns_info = group["name"]
                    return group["name"]
        return patterns_info


def set_config(config: LangKitConfig):
    global lang_config
    global regex_groups

    lang_config = config
    regex_groups = load_patterns()


# def set_config(config: LangKitConfig):
#     pattern_loader.set_config(config)
