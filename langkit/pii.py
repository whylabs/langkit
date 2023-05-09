from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import (
    register_metric_udf,
)
import json
import re
import os
import pkg_resources

pii_json_filename = "pii_regex.json"
json_path = pkg_resources.resource_filename(__name__, pii_json_filename)


@register_metric_udf(col_type=String)
def has_pii(text: str) -> str:
    pii_info = "No PII"
    # read json file
    with open(json_path, "r") as myfile:
        regex_groups = json.load(myfile)
    for group in regex_groups:
        for expression in group["expressions"]:
            if re.search(expression, text):
                pii_info = group["name"]
                return group["name"]
    return pii_info
