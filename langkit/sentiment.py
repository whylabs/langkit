from typing import Optional
from logging import getLogger

from whylogs.core.datatypes import String
from whylogs.experimental.core.udf_schema import register_type_udf

_lexicon = "vader_lexicon"
_sentiment_analyzer = None
_nltk_downloaded = False

diagnostic_logger = getLogger(__name__)

def init(lexicon: Optional[str] = None):
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    global _sentiment_analyzer, _nltk_downloaded
    if lexicon is None:
        lexicon = _lexicon
    if not _nltk_downloaded:
        nltk.download(lexicon)
        _nltk_downloaded = True
    _sentiment_analyzer = SentimentIntensityAnalyzer()


@register_type_udf(String, namespace="sentiment")
def sentiment_nltk(strings) -> list:
    if _sentiment_analyzer is None:
        raise ValueError(
            "sentiment metrics must initialize sentiment analyzer before evaluation!"
        )
    series_results = []
    for text in strings:
        try:
            sentiment_score = _sentiment_analyzer.polarity_scores(text)
            series_results.append(sentiment_score['compound'])
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in sentiment udf: {e}"
            )
            series_results.append(None)
    return series_results


init()
