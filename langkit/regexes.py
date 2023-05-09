from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import (
    register_metric_udf,
)
import json
import re
import pkg_resources
from typing import Optional
from logging import getLogger

logger = getLogger(__name__)

# If not stated otherwise, suggested patterns in the default file should reflect US patterns.
pattern_json_filename = "pattern_groups.json"
json_path = pkg_resources.resource_filename(__name__, pattern_json_filename)

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
    logger.warning(f"Could not find {pattern_json_filename}")
except json.decoder.JSONDecodeError:
    skip = True
    logger.warning(f"Could not parse {pattern_json_filename}")

if not skip:

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
