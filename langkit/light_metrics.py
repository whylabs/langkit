from typing import Optional
from whylogs.experimental.core.udf_schema import udf_schema
from whylogs.core.schema import DeclarativeSchema

from langkit import LangKitConfig
from langkit.metadata import attach_schema_metadata
from langkit import regexes
from langkit import textstat


def init(config: Optional[LangKitConfig] = None) -> DeclarativeSchema:
    regexes.init(config=config)
    textstat.init(config=config)

    text_schema = attach_schema_metadata(udf_schema(), "light_metrics")
    return text_schema
