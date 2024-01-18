from functools import partial

import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from langkit.core.metric import Metric, SingleMetric, SingleMetricResult, UdfInput


def sentiment_polarity_metric(column_name: str, lexicon: str = "vader_lexicon") -> Metric:
    # TODO Does this have built in idempotency?
    nltk.download(lexicon)  # type: ignore[reportUnknownMemberType]
    analyzer = SentimentIntensityAnalyzer()

    def udf(text: pd.DataFrame) -> SingleMetricResult:
        metrics = [analyzer.polarity_scores(t)["compound"] for t in UdfInput(text).iter_column_rows(column_name)]  # type: ignore[reportUnknownMemberType]
        return SingleMetricResult(metrics)

    return SingleMetric(
        name=f"{column_name}.sentiment_polarity",
        input_name=column_name,
        evaluate=udf,
    )


promp_sentiment_polarity = partial(sentiment_polarity_metric, "prompt")
response_sentiment_polarity = partial(sentiment_polarity_metric, "response")
promp_response_sentiment_polarity = [promp_sentiment_polarity, response_sentiment_polarity]
