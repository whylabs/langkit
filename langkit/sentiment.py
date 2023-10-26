from copy import deepcopy
from typing import Optional

from whylogs.experimental.core.udf_schema import register_dataset_udf
from . import LangKitConfig, lang_config, prompt_column, response_column


_prompt = prompt_column
_response = response_column
_sentiment_analyzer = None
_nltk_downloaded = False


def sentiment_nltk(text: str) -> float:
    if _sentiment_analyzer is None:
        raise ValueError(
            "sentiment metrics must initialize sentiment analyzer before evaluation!"
        )
    return _sentiment_analyzer.polarity_scores(text)["compound"]


def prompt_sentiment(text):
    return [sentiment_nltk(t) for t in text[_prompt]]


def response_sentiment(text):
    return [sentiment_nltk(t) for t in text[_response]]


def init(language: str = "en", lexicon: Optional[str] = None, config: Optional[LangKitConfig] = None):
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    config = config or deepcopy(lang_config)
    lexicon = lexicon or config.sentiment_lexicon
    global _sentiment_analyzer, _nltk_downloaded
    if not _nltk_downloaded:
        nltk.download(lexicon)
        _nltk_downloaded = True

    _sentiment_analyzer = SentimentIntensityAnalyzer()
    register_dataset_udf(
        [_prompt],
        udf_name=f"{language}.{_prompt}.sentiment_nltk",
        schema_name=language
    )(prompt_sentiment)
    register_dataset_udf(
        [_response],
        udf_name=f"{language}.{_response}.sentiment_nltk",
        schema_name=language
    )(response_sentiment)


init()
