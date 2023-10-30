from typing import Optional
from whylogs.experimental.core.udf_schema import udf_schema
from whylogs.core.schema import DeclarativeSchema

from . import LangKitConfig, multi_lang_config
from langkit import injections
from langkit import topics
from langkit import regexes
from langkit import sentiment
from langkit import textstat
from langkit import themes
from langkit import toxicity
from langkit import input_output


def init(language: str = "", config: Optional[LangKitConfig] = None) -> DeclarativeSchema:
    injections.init(language, config=config or multi_lang_config[language])
    topics.init(language, config=config or multi_lang_config[language])
    regexes.init(language, config=config or multi_lang_config[language])
    sentiment.init(language, config=config or multi_lang_config[language])
    textstat.init(language, config=config or multi_lang_config[language])
    themes.init(language, config=config or multi_lang_config[language])
    toxicity.init(language, config=config or multi_lang_config[language])
    input_output.init(language, config=config or multi_lang_config[language])
    text_schema = udf_schema()
    return text_schema
