from logging import getLogger
import pandas as pd
import whylogs as why
from whylogs.experimental.core.udf_schema import udf_schema

TEST_LOGGER = getLogger(__name__)


def test_vader_sentiment():
    from langkit import vader_sentiment

    vader_sentiment.init()
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
        dist = view.get_column(f"{column}.vader_sentiment").get_metric("distribution")
        assert "mean" in dist.to_summary_dict()
        TEST_LOGGER.debug(f"{column}.vader_sentiment has {dist.to_summary_dict()}")
        assert dist.avg > -0.4
        assert dist.min < 0
        assert dist.max >= 0


def test_vader_sentiment_with_llm_metrics():
    from langkit import llm_metrics
    from langkit import vader_sentiment

    vader_sentiment.init()
    schema = llm_metrics.init()
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
    view = why.log(df, schema=schema).view()
    TEST_LOGGER.debug(view.get_columns().keys())
    for column in ["prompt", "response"]:
        dist = view.get_column(f"{column}.vader_sentiment").get_metric("distribution")
        assert "mean" in dist.to_summary_dict()
        assert dist.min < 0
