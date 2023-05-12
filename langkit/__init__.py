from dataclasses import dataclass
import pkg_resources


@dataclass
class LangKitConfig:
    pattern_file_path: str = pkg_resources.resource_filename(
        __name__, "pattern_groups.json"
    )
    transformer_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    theme_file_path: str = pkg_resources.resource_filename(
        __name__, "themes.json"
    )
