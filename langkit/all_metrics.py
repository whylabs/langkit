from typing import Optional
from whylogs.experimental.core.udf_schema import udf_schema
from whylogs.core.schema import DeclarativeSchema

from langkit.metadata import attach_schema_metadata

from langkit import LangKitConfig, multi_lang_config
from langkit import injections
from langkit import topics
from langkit import regexes
from langkit import sentiment
from langkit import textstat
from langkit import themes
from langkit import toxicity
from langkit import input_output


def init(
    language: Optional[str] = None,
    config: Optional[LangKitConfig] = None,
    schema_name: str = "",
) -> DeclarativeSchema:
    injections.init(
        language, config=config or multi_lang_config[language], schema_name=schema_name
    )
    topics.init(
        language, config=config or multi_lang_config[language], schema_name=schema_name
    )
    regexes.init(
        language, config=config or multi_lang_config[language], schema_name=schema_name
    )
    sentiment.init(
        language, config=config or multi_lang_config[language], schema_name=schema_name
    )
    textstat.init(
        language, config=config or multi_lang_config[language], schema_name=schema_name
    )
    themes.init(
        language, config=config or multi_lang_config[language], schema_name=schema_name
    )
    toxicity.init(
        language, config=config or multi_lang_config[language], schema_name=schema_name
    )
    input_output.init(
        language, config=config or multi_lang_config[language], schema_name=schema_name
    )
    text_schema = attach_schema_metadata(
        udf_schema(schema_name=schema_name), "all_metrics"
    )
    return text_schema
