from typing import List, Optional
from whylogs.experimental.core.udf_schema import udf_schema
from whylogs.core.schema import DeclarativeSchema

from . import LangKitConfig
from langkit import injections
from langkit import topics
from langkit import regexes
from langkit import sentiment
from langkit import textstat
from langkit import themes
from langkit import toxicity
from langkit import input_output


def init(languages: List[str] = ["en"], config: Optional[LangKitConfig] = None) -> DeclarativeSchema:
    for language in langauges:
        injections.init(language, config=config)
        topics.init(language, config=config)
        regexes.init(language, config=config)
        sentiment.init(language, config=config)
        textstat.init(language, config=config)
        themes.init(language, config=config)
        toxicity.init(language, config=config)
        input_output.init(language, config=config)
    text_schema = udf_schema(chained_schemas=languages)
    return text_schema
