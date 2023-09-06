from typing import Optional
from whylogs.experimental.core.udf_schema import udf_schema
from whylogs.core.schema import DeclarativeSchema

from . import LangKitConfig
from langkit import regexes
from langkit import textstat


def init(config: Optional[LangKitConfig] = None) -> DeclarativeSchema:
    regexes.init(config=config)
    textstat.init(config=config)

    text_schema = udf_schema()
    return text_schema
