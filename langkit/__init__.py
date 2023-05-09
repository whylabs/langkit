from dataclasses import dataclass
import pkg_resources

pattern_json_filename = "pattern_groups.json"


@dataclass
class LangKitConfig:
    pattern_file_path: str = pkg_resources.resource_filename(
        __name__, pattern_json_filename
    )
