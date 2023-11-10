from langkit import __version__
from logging import getLogger

from typing import Any, Dict, Optional


_LANGKIT_VERSION_METADATA_KEY = "langkit.version"
_LANGKIT_METRIC_COLLECTION_KEY = "langkit.metric_collection"
diagnostic_logger = getLogger(__name__)


def _check_for_metadata(schema: Any) -> Optional[Dict[str, str]]:
    if schema is not None and hasattr(schema, "metadata"):
        metadata = getattr(schema, "metadata")
        if metadata is not None and isinstance(metadata, dict):
            return metadata
    return None


def _add_langkit_version_metadata(
    metadata: Dict[str, str], metric_collection_name: Optional[str]
) -> Dict[str, str]:
    if metadata is None:
        diagnostic_logger.warning("metadata is None, LangKit won't update metadata")
    else:
        metadata[_LANGKIT_VERSION_METADATA_KEY] = __version__
        if metric_collection_name:
            metadata[_LANGKIT_METRIC_COLLECTION_KEY] = metric_collection_name
    return metadata


def attach_schema_metadata(schema: Any, metric_collection_name: Optional[str]) -> Any:
    metadata = _check_for_metadata(schema)
    if metadata is None:
        diagnostic_logger.warning(
            "schema does not contain metadata, LangKit won't update metadata"
        )
        return schema
    _add_langkit_version_metadata(metadata, metric_collection_name)

    return schema
