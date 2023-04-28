import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import (
    register_metric_udf,
)

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()


@register_metric_udf(col_type=String)
def sentiment_nltk(text: str) -> float:
    sentiment_score = sia.polarity_scores(text)
    return sentiment_score['compound']
