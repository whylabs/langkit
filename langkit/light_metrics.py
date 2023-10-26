from typing import List, Optional
from whylogs.experimental.core.udf_schema import udf_schema
from whylogs.core.schema import DeclarativeSchema

from . import LangKitConfig
from langkit import regexes
from langkit import textstat


def init(languages: List[str] = ["en"], config: Optional[LangKitConfig] = None) -> DeclarativeSchema:
    for language in languages:
    regexes.init(language, config=config)
    textstat.init(language, config=config)

    text_schema = udf_schema(chained_schemas=languages)
    return text_schema
