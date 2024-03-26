from functools import lru_cache, partial

import nltk
import pandas as pd
from nltk.downloader import Downloader
from nltk.sentiment import SentimentIntensityAnalyzer

from langkit.core.metric import Metric, SingleMetric, SingleMetricResult, UdfInput


@lru_cache
def _get_analyzer():
    # Uses vader_lexicon by default, requires that its cached already
    return SentimentIntensityAnalyzer()


def sentiment_polarity_metric(column_name: str, lexicon: str = "vader_lexicon") -> Metric:
    def cache_assets():
        downloader = Downloader()
        if not downloader.is_installed(lexicon):  # pyright: ignore[reportUnknownMemberType]
            nltk.download(lexicon, raise_on_error=True)  # type: ignore[reportUnknownMemberType]

    def init():
        _get_analyzer()

    def udf(text: pd.DataFrame) -> SingleMetricResult:
        analyzer = _get_analyzer()
        metrics = [analyzer.polarity_scores(t)["compound"] for t in UdfInput(text).iter_column_rows(column_name)]  # type: ignore[reportUnknownMemberType]
        return SingleMetricResult(metrics)

    return SingleMetric(
        name=f"{column_name}.sentiment.sentiment_score",
        input_names=[column_name],
        evaluate=udf,
        init=init,
        cache_assets=cache_assets,
    )


prompt_sentiment_polarity = partial(sentiment_polarity_metric, "prompt")
response_sentiment_polarity = partial(sentiment_polarity_metric, "response")
prompt_response_sentiment_polarity = [prompt_sentiment_polarity, response_sentiment_polarity]
