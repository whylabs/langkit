import pandas as pd
from typing import Any, Dict, Optional
from whylogs.experimental.core.udf_schema import udf_schema


def extract(
    pandas: Optional[pd.DataFrame] = None,
    row: Optional[Dict[str, Any]] = None,
):
    schema = udf_schema()
    if pandas is not None:
        df_enhanced, _ = schema.apply_udfs(pandas=pandas)
        return df_enhanced
    if row is not None:
        _, row_enhanced = schema.apply_udfs(row=row)
        return row_enhanced
