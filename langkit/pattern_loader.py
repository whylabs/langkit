import json
import re
from logging import getLogger
from typing import Optional

from langkit import LangKitConfig, lang_config


diagnostic_logger = getLogger(__name__)


class PatternLoader:
    def __init__(self, json_path: Optional[str] = None):
        self.json_path = json_path
        self.regex_groups = self.load_patterns()

    def load_patterns(self):
        json_path = self.json_path
        if json_path is None:
            return None
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
                diagnostic_logger.info(f"Loaded regex pattern for {group['name']}")
        except FileNotFoundError:
            skip = True
            diagnostic_logger.warning(f"Could not find {json_path}")
        except json.decoder.JSONDecodeError as json_error:
            skip = True
            diagnostic_logger.warning(f"Could not parse {json_path}: {json_error}")
        return regex_groups if not skip else None

    def set_config(self, json_path: Optional[str] = None):
        self.json_path = json_path

    def update_patterns(self):
        self.regex_groups = self.load_patterns()

    def get_regex_groups(self):
        return self.regex_groups
