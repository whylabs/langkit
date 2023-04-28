from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import (
    register_metric_udf,
    _col_type_submetrics
)
from .textstat import *
from .nlp_udfs import *
from .sentiment import *
from .input_output import *


def register_text_metrics():
    return f"imported metrics for string features {_col_type_submetrics[String]}"
