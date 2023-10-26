from . import LangKitConfig
from logging import getLogger
from typing import List, Optional
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


def init(languages: List[str] = ["en"], config: Optional[LangKitConfig] = None) -> DeclarativeSchema:
    for language in languages:
        regexes.init(language, config=config)
        sentiment.init(language, config=config)
        textstat.init(language, config=config)
        themes.init(language, config=config)
        toxicity.init(language, config=config)
        input_output.init(language, config=config)

    text_schema = udf_schema(chained_schemas = languages)
    return text_schema
