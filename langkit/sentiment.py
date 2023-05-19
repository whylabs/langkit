import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import register_metric_udf

nltk.download("vader_lexicon")
_sentiment_analyzer = SentimentIntensityAnalyzer()


@register_metric_udf(col_type=String)
def sentiment_nltk(text: str) -> float:
    sentiment_score = _sentiment_analyzer.polarity_scores(text)
    return sentiment_score["compound"]
