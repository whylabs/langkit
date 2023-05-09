from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import (
    register_metric_udf,
)
import json
import re
from typing import Optional
from logging import getLogger
from . import LangKitConfig

logger = getLogger(__name__)

lang_config = LangKitConfig()


class PatternLoader:
    def __init__(self):
        self.config: LangKitConfig = LangKitConfig()
        self.regex_groups = self.load_patterns()

    def load_patterns(self):
        json_path = self.config.pattern_file_path
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

    def set_config(self, config: LangKitConfig):
        self.config = config

    def update_patterns(self):
        self.regex_groups = self.load_patterns()

    def get_regex_groups(self):
        return self.regex_groups


pattern_loader = PatternLoader()

if pattern_loader.get_regex_groups() is not None:

    @register_metric_udf(col_type=String)
    def has_patterns(text: str) -> Optional[str]:
        regex_groups = pattern_loader.get_regex_groups()
        patterns_info = ""
        for group in regex_groups:
            for expression in group["expressions"]:
                if expression.search(text):
                    patterns_info = group["name"]
                    return group["name"]
        return patterns_info


def set_config(config: LangKitConfig):
    pattern_loader.set_config(config)
    pattern_loader.update_patterns()
