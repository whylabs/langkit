from typing import Optional
from whylogs.experimental.core.udf_schema import udf_schema
from whylogs.core.schema import DeclarativeSchema

from langkit.metadata import attach_schema_metadata

from langkit import LangKitConfig
from langkit import injections
from langkit import topics
from langkit import regexes
from langkit import sentiment
from langkit import textstat
from langkit import themes
from langkit import toxicity
from langkit import input_output


def init(config: Optional[LangKitConfig] = None) -> DeclarativeSchema:
    injections.init(config=config)
    topics.init(config=config)
    regexes.init(config=config)
    sentiment.init(config=config)
    textstat.init(config=config)
    themes.init(config=config)
    toxicity.init(config=config)
    input_output.init(config=config)
    text_schema = attach_schema_metadata(udf_schema(), "all_metrics")
    return text_schema
