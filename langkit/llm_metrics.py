from langkit.metadata import attach_schema_metadata
from langkit import LangKitConfig
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


def init(config: Optional[LangKitConfig] = None) -> DeclarativeSchema:
    regexes.init(config=config)
    sentiment.init(config=config)
    textstat.init(config=config)
    themes.init(config=config)
    toxicity.init(config=config)
    input_output.init(config=config)

    text_schema = attach_schema_metadata(udf_schema(), "llm_metrics")
    return text_schema
