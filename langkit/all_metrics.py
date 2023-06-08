from whylogs.experimental.core.udf_schema import udf_schema
from whylogs.core.schema import DeclarativeSchema

from langkit import injections
from langkit import topics
from langkit import regexes
from langkit import sentiment
from langkit import textstat
from langkit import themes
from langkit import toxicity
from langkit import input_output


def init() -> DeclarativeSchema:
    injections.init()
    topics.init()
    regexes.init()
    sentiment.init()
    textstat.init()
    themes.init()
    toxicity.init()
    input_output.init()
    text_schema = udf_schema()
    return text_schema
