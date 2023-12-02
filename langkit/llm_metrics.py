from langkit.metadata import attach_schema_metadata
from langkit import LangKitConfig, multi_lang_config
from logging import getLogger
from typing import Optional
from whylogs.experimental.core.udf_schema import udf_schema
from whylogs.core.schema import DeclarativeSchema

diagnostic_logger = getLogger(__name__)

try:
    from langkit import regexes
    from langkit import sentiment
    from langkit import textstat
    from langkit import themes
    from langkit import toxicity
    from langkit import input_output
except ImportError:
    raise ImportError(
        "To use `llm_metrics` please install it with `pip install langkit[all]`."
    )


def init(
    language: Optional[str] = None,
    config: Optional[LangKitConfig] = None,
    schema_name: str = "",
) -> DeclarativeSchema:
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
        udf_schema(schema_name=schema_name), "llm_metrics"
    )
    return text_schema
