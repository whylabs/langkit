from copy import deepcopy
from typing import Optional

from whylogs.experimental.core.udf_schema import register_dataset_udf
from langkit import LangKitConfig, lang_config, prompt_column, response_column


_prompt = prompt_column
_response = response_column
_sentiment_analyzer = None


def vader_sentiment(text: str) -> float:
    if _sentiment_analyzer is None:
        raise ValueError(
            "sentiment metrics must initialize sentiment analyzer before evaluation!"
        )
    return _sentiment_analyzer.polarity_scores(text)["compound"]


@register_dataset_udf([_prompt], udf_name=f"{_prompt}.vader_sentiment")
def prompt_sentiment(text):
    return [vader_sentiment(t) for t in text[_prompt]]


@register_dataset_udf([_response], udf_name=f"{_response}.vader_sentiment")
def response_sentiment(text):
    return [vader_sentiment(t) for t in text[_response]]


def init(lexicon: Optional[str] = None, config: Optional[LangKitConfig] = None):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    config = config or deepcopy(lang_config)
    global _sentiment_analyzer
    _sentiment_analyzer = SentimentIntensityAnalyzer()
