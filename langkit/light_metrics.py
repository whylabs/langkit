from whylogs.experimental.core.udf_schema import udf_schema
from whylogs.core.schema import DeclarativeSchema

from langkit import regexes
from langkit import textstat


def init() -> DeclarativeSchema:
    regexes.init()
    textstat.init()

    text_schema = udf_schema()
    return text_schema
