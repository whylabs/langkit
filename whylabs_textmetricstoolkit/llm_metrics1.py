from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import (
    register_metric_udf,
)
import textstat

#Here's how you can wire a udf, just add the decorator like these three examples

@register_metric_udf(col_type=String)
def flesch_kincaid_grade(text: str) -> float:
    # TODO: check if input is actually a string, try/except
    return textstat.flesch_kincaid_grade(text)

@register_metric_udf(col_type=String)
def smog_index(text: str) -> float:
    # TODO: check if input is actually a string, try/except
    return textstat.smog_index(text)

@register_metric_udf(col_type=String)
def dale_chall_readability_score(text: str) -> float:
    # TODO: check if input is actually a string, try/except
    return textstat.dale_chall_readability_score(text)