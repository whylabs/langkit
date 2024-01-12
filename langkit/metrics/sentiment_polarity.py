from functools import partial

import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from langkit.metrics.metric import Metric, MetricResult, UdfInput


def __sentiment_polarity_module(column_name: str, lexicon: str = "vader_lexicon") -> Metric:
    # TODO Does this have built in idempotency?
    nltk.download(lexicon)  # type: ignore[reportUnknownMemberType]
    analyzer = SentimentIntensityAnalyzer()

    def udf(text: pd.DataFrame) -> MetricResult:
        metrics = [analyzer.polarity_scores(t)["compound"] for t in UdfInput(text).iter_column_rows(column_name)]  # type: ignore[reportUnknownMemberType]
        return MetricResult(metrics)

    return Metric(
        name=f"{column_name}.sentiment_polarity",
        input_name=column_name,
        evaluate=udf,
    )


promp_sentiment_polarity = partial(__sentiment_polarity_module, "prompt")
response_sentiment_polarity = partial(__sentiment_polarity_module, "response")
promp_response_sentiment_polarity = [promp_sentiment_polarity, response_sentiment_polarity]
