from typing import Optional
from whylogs.experimental.core.udf_schema import udf_schema
from whylogs.core.schema import DeclarativeSchema

from langkit import LangKitConfig, multi_lang_config
from langkit.metadata import attach_schema_metadata
from langkit import regexes
from langkit import textstat


def init(
    language: Optional[str] = None,
    config: Optional[LangKitConfig] = None,
    schema_name: str = "",
) -> DeclarativeSchema:
    regexes.init(
        language, config=config or multi_lang_config[language], schema_name=schema_name
    )
    textstat.init(
        language, config=config or multi_lang_config[language], schema_name=schema_name
    )

    text_schema = attach_schema_metadata(
        udf_schema(schema_name=schema_name), "light_metrics"
    )
    return text_schema
