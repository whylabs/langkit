import pandas as pd
from typing import Any, Dict, Optional, Union
from whylogs.experimental.core.udf_schema import udf_schema, UdfSchema


def extract(
    data: Union[pd.DataFrame, Dict[str, Any]],
    schema: Optional[UdfSchema] = None,
):
    if schema is None:
        schema = udf_schema()
    if isinstance(data, pd.DataFrame):
        df_enhanced, _ = schema.apply_udfs(pandas=data)
        return df_enhanced
    elif isinstance(data, dict):
        _, row_enhanced = schema.apply_udfs(row=data)
        return row_enhanced
    raise ValueError(
        f"Extract: data of type {type(data)} is invalid: supported input types are pandas dataframe or dictionary"
    )
