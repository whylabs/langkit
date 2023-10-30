from . import LangKitConfig, multi_lang_config
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
    language: str = "", config: Optional[LangKitConfig] = None
) -> DeclarativeSchema:
    regexes.init(language, config=config or multi_lang_config[language])
    sentiment.init(language, config=config or multi_lang_config[language])
    textstat.init(language, config=config or multi_lang_config[language])
    themes.init(language, config=config or multi_lang_config[language])
    toxicity.init(language, config=config or multi_lang_config[language])
    input_output.init(language, config=config or multi_lang_config[language])

    text_schema = udf_schema()
    return text_schema
