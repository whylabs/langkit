import langkit
import pandas as pd

row = {"prompt": "I love you", "response": "I hate you"}
df = pd.DataFrame({"prompt": ["I love you", "I hate you"]})


# toxicity.init()
# sentiment.init()
def test_extract_pandas():
    from langkit import textstat

    textstat.init()
    df = pd.DataFrame({"prompt": ["I love you", "I hate you"]})
    enhanced_df = langkit.extract(pandas=df)
    assert "prompt.flesch_reading_ease" in enhanced_df.columns


def test_extract_row():
    from langkit import regexes

    regexes.init()
    row = {"prompt": "I love you", "response": "address: 123 Main St."}
    enhanced_row = langkit.extract(row=row)
    assert enhanced_row.get("response.has_patterns") == "mailing address"
    assert not enhanced_row.get("prompt.has_patterns")


def test_extract_light_metrics():
    from langkit import light_metrics

    light_metrics.init()

    row = {"prompt": "I love you", "response": "address: 123 Main St."}
    enhanced_row = langkit.extract(row=row)
    assert enhanced_row.get("response.has_patterns") == "mailing address"
    assert not enhanced_row.get("prompt.has_patterns")
    assert "prompt.flesch_reading_ease" in enhanced_row.keys()
