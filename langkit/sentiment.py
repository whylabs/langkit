from copy import deepcopy
from typing import Optional, Set

from whylogs.experimental.core.udf_schema import register_dataset_udf
from . import LangKitConfig, lang_config, prompt_column, response_column
from langkit.whylogs.unreg import unregister_udfs


def sentiment_nltk(text: str, sentiment_analyzer) -> float:
    if sentiment_analyzer is None:
        raise ValueError(
            "sentiment metrics must initialize sentiment analyzer before evaluation!"
        )
    return sentiment_analyzer.polarity_scores(text)["compound"]


_supported_languages = {
    "ar",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "id",
    "it",
    "ja",
    "ms",
    "pt",
    "zh",
}


def sentiment_multilingual(text: str, pipeline) -> float:
    if pipeline is None:
        raise ValueError("sentiment score must initialize the pipeline first")

    result = pipeline(text, return_all_scores=True)
    return [res["score"] for res in result[0] if res["label"] == "positive"][0]


def _sentiment_wrapper(sentiment_fn, argument, column):
    def _wrappee(text):
        return [sentiment_fn(t, argument) for t in text[column]]

    return _wrappee


_registered: Set[str] = set()


_nltk_downloaded = None
_response_nltk_downloaded = None
_sentiment_analyzer = None
_response_sentiment_analyzer = None


def configure_nltk(config, lexicon, response_lexicon):
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    lexicon = lexicon or config.sentiment_lexicon
    global _nltk_downloaded, _sentiment_analyzer
    if _nltk_downloaded != lexicon:
        nltk.download(lexicon)
        _nltk_downloaded = lexicon
        _sentiment_analyzer = (
            SentimentIntensityAnalyzer()
        )  # TODO: probably need to pass an argument
    else:
        _sentiment_analyzer = None

    lexicon = response_lexicon or config.response_sentiment_lexicon
    global _response_nltk_downloaded, _response_sentiment_analyzer
    if _response_nltk_downloaded != lexicon:
        nltk.download(lexicon)
        _response_nltk_downloaded = lexicon
        _response_sentiment_analyzer = (
            SentimentIntensityAnalyzer()
        )  # TODO: needs argument
    else:
        _response_sentiment_analyzer = None


_pipeline = None
_response_pipeline = None


def configure_hugging_face(config, sentiment_model_path, response_sentiment_model_path):
    from transformers import pipeline

    global _pipeline, _response_pipeline
    model_path = sentiment_model_path or config.sentiment_model_path
    if model_path:
        _pipeline = pipeline(model=model_path, top_k=None)
    else:
        _pipeline = None

    model_path = response_sentiment_model_path or config.response_sentiment_model_path
    if model_path:
        _response_pipeline = pipeline(model=model_path, top_k=None)
    else:
        _response_pipeline = None


def init(
    language: Optional[str] = None,
    lexicon: Optional[str] = None,
    config: Optional[LangKitConfig] = None,
    response_lexicon: Optional[str] = None,
    sentiment_model_path: Optional[str] = None,
    response_sentiment_model_path: Optional[str] = None,
):
    global _registered
    unregister_udfs(_registered)

    config = config or deepcopy(lang_config)
    prompt_languages = {language} if language is not None else config.prompt_languages
    response_languages = (
        {language} if language is not None else config.response_languages
    )

    configure_nltk(config, lexicon, response_lexicon)
    configure_hugging_face(config, sentiment_model_path, response_sentiment_model_path)

    if prompt_languages is not None and len(prompt_languages) > 0:
        if prompt_languages.issubset({"", "en"}) and _sentiment_analyzer:
            register_dataset_udf(
                [prompt_column], udf_name=f"{prompt_column}.sentiment_nltk"
            )(_sentiment_wrapper(sentiment_nltk, _sentiment_analyzer, prompt_column))
            _registered.add(f"{prompt_column}.sentiment_nltk")
        elif prompt_languages.issubset(_supported_languages) and _pipeline:
            register_dataset_udf(
                [prompt_column], udf_name=f"{prompt_column}.sentiment_multi"
            )(_sentiment_wrapper(sentiment_multilingual, _pipeline, prompt_column))
            _registered.add(f"{prompt_column}.sentiment_multi")

    if response_languages is not None and len(response_languages) > 0:
        if response_languages.issubset({"", "en"}) and _response_sentiment_analyzer:
            register_dataset_udf(
                [response_column], udf_name=f"{response_column}.sentiment_nltk"
            )(
                _sentiment_wrapper(
                    sentiment_nltk, _response_sentiment_analyzer, response_column
                )
            )
            _registered.add(f"{response_column}.sentiment_nltk")
        elif response_languages.issubset(_supported_languages) and _response_pipeline:
            register_dataset_udf(
                [prompt_column], udf_name=f"{response_column}.sentiment_multi"
            )(
                _sentiment_wrapper(
                    sentiment_multilingual, _response_pipeline, prompt_column
                )
            )
            _registered.add(f"{response_column}.sentiment_multi")
