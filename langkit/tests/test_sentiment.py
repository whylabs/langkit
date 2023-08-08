import pandas as pd
import pytest
import whylogs as why
from whylogs.experimental.core.udf_schema import udf_schema


@pytest.mark.load
def test_sentiment():
    import langkit.sentiment

    langkit.sentiment.init()
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
    schema = udf_schema()
    view = why.log(df, schema=schema).view()
    print(view.get_columns().keys())
    for column in ["prompt", "response"]:
        dist = (
            view.get_column(f"{column}.sentiment_nltk")
            .get_metric("distribution")
            .to_summary_dict()
        )
        assert "mean" in dist
