import pandas as pd
import os
import whylogs as why
from whylogs.core.resolvers import STANDARD_RESOLVER
from whylogs.core.schema import DeclarativeSchema

from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import register_metric_udf

from langdetect import detect
import octopii_text_utils as text_utils

@register_metric_udf(col_type=String)
def detect_language(text: str) -> str:
    return detect(text)

@register_metric_udf(col_type=String)
def get_pii_score(text: str) -> int:
    intelligible = text_utils.string_tokenizer(text)
    keywords_scores = text_utils.keywords_classify_pii(intelligible)
    score = max(keywords_scores.values())
    return score

@register_metric_udf(col_type=String)
def get_pii_class(text: str) -> str:
    intelligible = text_utils.string_tokenizer(text)
    keywords_scores = text_utils.keywords_classify_pii(intelligible)
    score = max(keywords_scores.values())
    pii_class = list(keywords_scores.keys())[list(keywords_scores.values()).index(score)]
    return pii_class

@register_metric_udf(col_type=String)
def get_emails_count(text: str) -> int:
    emails = text_utils.email_pii(text)
    return len(emails)
