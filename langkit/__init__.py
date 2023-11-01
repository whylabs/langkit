from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import importlib.resources as resources


def _resource_filename(file_name):
    with resources.path(__package__, file_name) as path:
        return str(path)


@dataclass
class LangKitConfig:
    pattern_file_path: str = field(
        default_factory=lambda: _resource_filename("pattern_groups.json")
    )
    response_pattern_file_path: Optional[str] = field(
        default_factory=lambda: _resource_filename("pattern_groups.json")
    )
    metric_name_map: Dict[str, str] = field(default_factory=dict)
    theme_file_path: str = field(
        default_factory=lambda: _resource_filename("themes.json")
    )
    response_theme_file_path: str = field(
        default_factory=lambda: _resource_filename("themes.json")
    )
    transformer_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"
    response_transformer_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"
    topics: List[str] = field(
        default_factory=lambda: [
            "law",
            "finance",
            "medical",
            "education",
            "politics",
            "support",
        ]
    )
    response_topics: List[str] = field(
        default_factory=lambda: [
            "law",
            "finance",
            "medical",
            "education",
            "politics",
            "support",
        ]
    )
    nlp_scores: list = field(
        default_factory=lambda: [
            "bleu",
            "rouge",
            "meteor",
        ]
    )
    reference_corpus: Optional[str] = ""
    injections_base_url = (
        "https://whylabs-public.s3.us-west-2.amazonaws.com/langkit/data/injections/"
    )
    data_folder: str = "langkit_data"
    rouge_type: str = "rouge1"
    sentiment_lexicon: Optional[str] = "vader_lexicon"
    response_sentiment_lexicon: Optional[str] = "vader_lexicon"
    topic_model_path: Optional[
        str
    ] = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    response_topic_model_path: Optional[
        str
    ] = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    topic_classifier: Optional[str] = "zero-shot-classification"
    response_topic_classifier: Optional[str] = "zero-shot-classification"
    toxicity_model_path: Optional[str] = "martin-ha/toxic-comment-model"
    response_toxicity_model_path: Optional[str] = "martin-ha/toxic-comment-model"
    injections_transformer_name: Optional[str] = "all-MiniLM-L6-v2"
    injections_version: Optional[str] = "v1"
    prompt_languages: Optional[Set[str]] = field(default_factory=lambda: {"en"})
    response_languages: Optional[Set[str]] = field(default_factory=lambda: {"en"})


prompt_column: str = "prompt"
response_column: str = "response"
lang_config = LangKitConfig()


# Override default models/parameters per language
multi_lang_config: Dict[Optional[str], LangKitConfig] = {
    None: LangKitConfig(),
    "": LangKitConfig(),
    "ar": LangKitConfig(
        prompt_languages={"ar"},
        response_languages={"ar"},
        injections_transformer_name=None,
        reference_corpus=None,
        sentiment_lexicon=None,
        response_sentiment_lexicon=None,
        topic_model_path=None,
        response_topic_model_path=None,
        toxicity_model_path=None,
        response_toxicity_model_path=None,
        transformer_name=None,
        response_transformer_name=None,
    ),
    "en": LangKitConfig(),
    "es": LangKitConfig(
        prompt_languages={"es"},
        response_languages={"es"},
        injections_transformer_name=None,
        reference_corpus=None,
        sentiment_lexicon=None,
        response_sentiment_lexicon=None,
        topic_model_path=None,
        response_topic_model_path=None,
        toxicity_model_path=None,
        response_toxicity_model_path=None,
        transformer_name=None,
        response_transformer_name=None,
    ),
    "it": LangKitConfig(
        prompt_languages={"it"},
        response_languages={"it"},
        injections_transformer_name=None,
        reference_corpus=None,
        sentiment_lexicon=None,
        response_sentiment_lexicon=None,
        topic_model_path=None,
        response_topic_model_path=None,
        toxicity_model_path=None,
        response_toxicity_model_path=None,
        transformer_name=None,
        response_transformer_name=None,
    ),
    "pt": LangKitConfig(
        prompt_languages={"pt"},
        response_languages={"pt"},
        injections_transformer_name=None,
        reference_corpus=None,
        sentiment_lexicon=None,
        response_sentiment_lexicon=None,
        topic_model_path=None,
        response_topic_model_path=None,
        toxicity_model_path="dougtrajano/toxicity-type-detection",
        response_toxicity_model_path="dougtrajano/toxicity-type-detection",
        transformer_name=None,
        response_transformer_name=None,
    ),
}


def package_version(package: str = __package__) -> str:
    """Calculate version number based on pyproject.toml"""
    try:
        from importlib import metadata

        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = f"{package} is not installed."

    return version


__version__ = package_version()

__ALL__ = [__version__, LangKitConfig]
