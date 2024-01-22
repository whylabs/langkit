from dataclasses import dataclass, field
from typing import Dict, List
from .extract import extract
import importlib.resources as resources


def _resource_filename(file_name):
    with resources.path(__package__, file_name) as path:
        return str(path)


@dataclass
class LangKitConfig:
    pattern_file_path: str = field(
        default_factory=lambda: _resource_filename("pattern_groups.json")
    )
    pii_entities_file_path: str = field(
        default_factory=lambda: _resource_filename("PII_entities.json")
    )
    metric_name_map: Dict[str, str] = field(default_factory=dict)
    theme_file_path: str = field(
        default_factory=lambda: _resource_filename("themes.json")
    )
    transformer_name: str = "sentence-transformers/all-MiniLM-L6-v2"
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
    nlp_scores: list = field(
        default_factory=lambda: [
            "bleu",
            "rouge",
            "meteor",
        ]
    )
    reference_corpus: str = ""
    injections_base_url = (
        "https://whylabs-public.s3.us-west-2.amazonaws.com/langkit/data/injections/"
    )
    data_folder: str = "langkit_data"
    rouge_type: str = "rouge1"
    sentiment_lexicon: str = "vader_lexicon"
    topic_model_path: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    topic_classifier: str = "zero-shot-classification"
    toxicity_model_path: str = "martin-ha/toxic-comment-model"


prompt_column: str = "prompt"
response_column: str = "response"
lang_config = LangKitConfig()


def package_version(package: str = __package__) -> str:
    """Calculate version number based on pyproject.toml"""
    try:
        from importlib import metadata

        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = f"{package} is not installed."

    return version


__version__ = package_version()

__ALL__ = [__version__, LangKitConfig, extract]
