from logging import getLogger
import pandas as pd
import whylogs as why
from whylogs.experimental.core.udf_schema import udf_schema

TEST_LOGGER = getLogger(__name__)

def test_vader_sentiment():
    import langkit.vader_sentiment as sentiment

    sentiment.init()
    df = pd.DataFrame(
        {
            "prompt": [
                "timely text propagated prolifically",
                "articulate artichokes aggravated allergically",
                "So amazing!! ABSOLUTELY fantastic :-) Da BOMB!!",
            ],
            "response": [
                "I neither approve or disapprove.",
                "strawberries are not true fruits, and they smell",
                "this is not my response",
            ],
        }
    )
    schema = udf_schema()
    view = why.log(df, schema=schema).view()
    print(view.get_columns().keys())
    for column in ["prompt", "response"]:
        dist = (
            view.get_column(f"{column}.vader_sentiment")
            .get_metric("distribution")
            .to_summary_dict()
        )
        assert "mean" in dist
        TEST_LOGGER.info(view.get_column(f"{column}.vader_sentiment")
            .get_metric("distribution")
            .to_summary_dict())
