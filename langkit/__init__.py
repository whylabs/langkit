from dataclasses import dataclass, field

import importlib.resources as resources


def _resource_filename(file_name):
    with resources.path(__package__, file_name) as path:
        return str(path)


@dataclass
class LangKitConfig:
    pattern_file_path: str = field(
        default_factory=lambda: _resource_filename("pattern_groups.json")
    )
    theme_file_path: str = field(
        default_factory=lambda: _resource_filename("themes.json")
    )
    transformer_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    topics: list = field(
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

__ALL__ = [__version__, LangKitConfig]
