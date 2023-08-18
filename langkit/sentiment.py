from typing import Optional

from whylogs.experimental.core.udf_schema import register_dataset_udf
from . import prompt_column, response_column


_prompt = prompt_column
_response = response_column
_lexicon = "vader_lexicon"
_sentiment_analyzer = None
_nltk_downloaded = False


def sentiment_nltk(text: str) -> float:
    if _sentiment_analyzer is None:
        raise ValueError(
            "sentiment metrics must initialize sentiment analyzer before evaluation!"
        )
    return _sentiment_analyzer.polarity_scores(text)["compound"]


@register_dataset_udf([_prompt], udf_name=f"{_prompt}.sentiment_nltk")
def prompt_sentiment(text):
    return [sentiment_nltk(t) for t in text[_prompt]]


@register_dataset_udf([_response], udf_name=f"{_response}.sentiment_nltk")
def response_sentiment(text):
    return [sentiment_nltk(t) for t in text[_response]]


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


init()
