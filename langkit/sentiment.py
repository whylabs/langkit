from typing import Optional

from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import register_metric_udf

_lexicon = "vader_lexicon"
_sentiment_analyzer = None
_nltk_download_count = 0


def init(lexicon: Optional[str] = None):
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    global _sentiment_analyzer, _nltk_download_count
    if lexicon is None:
        lexicon = _lexicon
    if _nltk_download_count == 0:
        nltk.download(lexicon)
        _nltk_download_count = 1
    _sentiment_analyzer = SentimentIntensityAnalyzer()


@register_metric_udf(col_type=String)
def sentiment_nltk(text: str) -> float:
    if _sentiment_analyzer is None:
        raise ValueError(
            "sentiment metrics must initialize sentiment analyzer before evaluation!"
        )
    sentiment_score = _sentiment_analyzer.polarity_scores(text)
    return sentiment_score["compound"]


init()
