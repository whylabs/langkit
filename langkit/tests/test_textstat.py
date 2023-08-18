import pandas as pd
import langkit.textstat as ts
import whylogs as why
from whylogs.experimental.core.udf_schema import udf_schema


def _unpack(tuples):
    return [(t[2], t[1]) if len(t) == 3 else (t[0], t[1]) for t in tuples]


def test_textstat():
    df = pd.DataFrame(
        {
            "prompt": [
                "timely text propagated prolifically",
                "articulate artichokes aggravated allergically",
                "just some words",
            ],
            "response": [
                "words words everywhere but not a thought to think",
                "strawberries are not true fruits",
                "this is not my response",
            ],
        }
    )
    udf_list = _unpack(ts._udfs_to_register) + [
        ("aggregate_reading_level", ""),
    ]
    schema_names = set([s for _, s in udf_list])
    for schema_name in schema_names:
        schema = udf_schema(schema_name=schema_name)
        view = why.log(df, schema=schema).view()
        for stat, stat_schema in udf_list:
            for column in ["prompt", "response"]:
                if stat_schema in {"", schema_name}:
                    dist = (
                        view.get_column(f"{column}.{stat}")
                        .get_metric("distribution")
                        .to_summary_dict()
                    )
                    assert "mean" in dist
                else:
                    assert view.get_column(f"{column}.{stat}") is None
