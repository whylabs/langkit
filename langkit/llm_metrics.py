from logging import getLogger
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


def init() -> DeclarativeSchema:
    regexes.init()
    sentiment.init()
    textstat.init()
    themes.init()
    toxicity.init()
    input_output.init()

    text_schema = udf_schema()
    return text_schema
