from typing import Optional

from whylogs.core.stubs import pd
from whylogs.experimental.core.udf_schema import register_dataset_udf
from . import LangKitConfig

lang_config = LangKitConfig()
_lexicon = "vader_lexicon"
_sentiment_analyzer = None
_nltk_downloaded = False


def sentiment_nltk(text):
    if _sentiment_analyzer is None:
        raise ValueError(
            "sentiment metrics must initialize sentiment analyzer before evaluation!"
        )
    result = []
    index = text.columns[0] if isinstance(text, pd.DataFrame) else list(text.keys())[0]
    for input in text[index]:
        result.append(_sentiment_analyzer.polarity_scores(input)["compound"])
    return result


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
    for column in [lang_config.prompt_column, lang_config.response_column]:
        register_dataset_udf([column], udf_name=f"{column}.sentiment_nltk")(
            sentiment_nltk
        )


init()
