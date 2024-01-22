from logging import getLogger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from whylogs.experimental.core.udf_schema import register_dataset_udf
from langkit import prompt_column, response_column


_prompt = prompt_column
_response = response_column
_vader_sentiment_analyzer = None
diagnostic_logger = getLogger(__name__)


def vader_sentiment(text: str) -> float:
    global _vader_sentiment_analyzer
    if _vader_sentiment_analyzer is None:
        diagnostic_logger.info(
            "vader_sentiment called before init, using default initialization."
        )
        _vader_sentiment_analyzer = init()
    return _vader_sentiment_analyzer.polarity_scores(text)["compound"]


@register_dataset_udf([_prompt], udf_name=f"{_prompt}.vader_sentiment")
def prompt_sentiment(text):
    return [vader_sentiment(t) for t in text[_prompt]]


@register_dataset_udf([_response], udf_name=f"{_response}.vader_sentiment")
def response_sentiment(text):
    return [vader_sentiment(t) for t in text[_response]]


def init() -> SentimentIntensityAnalyzer:
    global _vader_sentiment_analyzer
    _vader_sentiment_analyzer = SentimentIntensityAnalyzer()
    return _vader_sentiment_analyzer
