from typing import Optional
from whylogs.experimental.core.udf_schema import udf_schema
from whylogs.core.schema import DeclarativeSchema

from . import LangKitConfig, multi_lang_config
from langkit import regexes
from langkit import textstat


def init(
    language: str = "", config: Optional[LangKitConfig] = None
) -> DeclarativeSchema:
    regexes.init(language, config=config or multi_lang_config[language])
    textstat.init(language, config=config or multi_lang_config[language])

    text_schema = udf_schema()
    return text_schema
