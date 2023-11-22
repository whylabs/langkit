import langkit
import pandas as pd
from whylogs.experimental.core.udf_schema import UdfSchema, UdfSpec


def test_extract_pandas():
    from langkit import textstat

    textstat.init()
    df = pd.DataFrame({"prompt": ["I love you", "I hate you"]})
    enhanced_df = langkit.extract(data=df)
    assert "prompt.flesch_reading_ease" in enhanced_df.columns


def test_extract_row():
    from langkit import regexes

    regexes.init()
    row = {"prompt": "I love you", "response": "address: 123 Main St."}
    enhanced_row = langkit.extract(data=row)
    assert enhanced_row.get("response.has_patterns") == "mailing address"
    assert not enhanced_row.get("prompt.has_patterns")


def test_extract_light_metrics():
    from langkit import light_metrics

    light_metrics.init()

    row = {"prompt": "I love you", "response": "address: 123 Main St."}
    enhanced_row = langkit.extract(row)
    assert enhanced_row.get("response.has_patterns") == "mailing address"
    assert not enhanced_row.get("prompt.has_patterns")
    assert "prompt.flesch_reading_ease" in enhanced_row.keys()


def test_extract_with_custom_schema():
    schema = UdfSchema(
        udf_specs=[
            UdfSpec(
                column_names=["prompt"],
                udfs={"prompt.customfeature": lambda x: x["prompt"]},
            )
        ],
    )
    row = {"prompt": "I love you", "response": "address: 123 Main St."}
    enhanced_row = langkit.extract(row, schema=schema)
    assert enhanced_row.get("prompt.customfeature") == "I love you"
