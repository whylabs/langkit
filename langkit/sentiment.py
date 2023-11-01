from copy import deepcopy
from typing import Optional

from whylogs.experimental.core.udf_schema import register_dataset_udf
from . import LangKitConfig, lang_config, prompt_column, response_column


_nltk_downloaded = None
_response_nltk_downloaded = None


def sentiment_nltk(text: str, sentiment_analyzer) -> float:
    if sentiment_analyzer is None:
        raise ValueError(
            "sentiment metrics must initialize sentiment analyzer before evaluation!"
        )
    return sentiment_analyzer.polarity_scores(text)["compound"]


def _sentiment_wrapper(sentiment_analyzer, column):
    def _wrappee(text):
        return [sentiment_nltk(t, sentiment_analyzer) for t in text[column]]

    return _wrappee


def init(
    language: Optional[str] = None,
    lexicon: Optional[str] = None,
    config: Optional[LangKitConfig] = None,
    response_lexicon: Optional[str] = None,
):
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    config = config or deepcopy(lang_config)
    lexicon = lexicon or config.sentiment_lexicon
    global _nltk_downloaded
    if _nltk_downloaded != lexicon:
        nltk.download(lexicon)
        _nltk_downloaded = lexicon
        _sentiment_analyzer = (
            SentimentIntensityAnalyzer()
        )  # TODO: probably need to pass an argument
        register_dataset_udf(
            [prompt_column], udf_name=f"{prompt_column}.sentiment_nltk"
        )(_sentiment_wrapper(_sentiment_analyzer, prompt_column))
    else:
        _sentiment_analyzer = None

    lexicon = response_lexicon or config.response_sentiment_lexicon
    global _response_nltk_downloaded
    if _response_nltk_downloaded != lexicon:
        nltk.download(lexicon)
        _response_nltk_downloaded = lexicon
        _response_sentiment_analyzer = (
            SentimentIntensityAnalyzer()
        )  # TODO: needs argument
        register_dataset_udf(
            [response_column], udf_name=f"{response_column}.sentiment_nltk"
        )(_sentiment_wrapper(_response_sentiment_analyzer, response_column))
    else:
        _sentiment_analyzer = None
