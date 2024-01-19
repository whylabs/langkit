from copy import deepcopy
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from whylogs.experimental.core.udf_schema import (
    register_multioutput_udf,
)
import pandas as pd
from typing import Dict, List, Optional, Tuple
from langkit import LangKitConfig, lang_config, prompt_column, response_column
from langkit.pattern_loader import PresidioEntityLoader
from langkit.utils import _unregister_metric_udf
import json

_registered: List[str] = []

entity_loader = PresidioEntityLoader()


# entities = ["PHONE_NUMBER", "US_PASSPORT"]
analyzer = AnalyzerEngine()


def format_presidio_result(result: RecognizerResult) -> dict:
    return {
        "type": f"{result.entity_type}",
        "start": f"{result.start}",
        "end": f"{result.end}",
        "score": f"{result.score}",
    }


def analyze_pii(text: str) -> Tuple[str, int]:
    global analyzer
    global entity_loader

    entities = entity_loader.get_entities()
    results = analyzer.analyze(
        text=text,
        entities=entities,
        language="en",
    )
    dict_results = [format_presidio_result(entity) for entity in results]
    return (json.dumps(dict_results), len(dict_results))


def _wrapper(column):
    def wrappee(text):
        analyzer_results: List[tuple] = []
        for input in text[column]:
            analyzer_results.append(analyze_pii(input))
        to_return = {
            "result": [x[0] for x in analyzer_results],
            "entities_count": [x[1] for x in analyzer_results],
        }
        if isinstance(text, pd.DataFrame):
            return pd.DataFrame(to_return)
        else:
            return to_return

    return wrappee


def _register_udfs(config: Optional[LangKitConfig] = None):
    from whylogs.experimental.core.udf_schema import _resolver_specs

    global _registered
    if _registered and config is None:
        return
    if config is None:
        config = lang_config
    default_metric_name = "pii_presidio"
    entity_metric_name = config.metric_name_map.get(
        default_metric_name, default_metric_name
    )

    for old in _registered:
        _unregister_metric_udf(old_name=old)
        if (
            _resolver_specs is not None
            and isinstance(_resolver_specs, Dict)
            and isinstance(_resolver_specs[""], List)
        ):
            _resolver_specs[""] = [
                spec for spec in _resolver_specs[""] if spec.column_name != old
            ]
    _registered = []

    if entity_loader.get_entities() is not None:
        for column in [prompt_column, response_column]:
            udf_name = f"{column}.{entity_metric_name}"
            register_multioutput_udf(
                [column],
                prefix=udf_name,
            )(_wrapper(column))
            _registered.append(udf_name)


def init(
    entities_file_path: Optional[str] = None, config: Optional[LangKitConfig] = None
):
    config = deepcopy(config or lang_config)
    if entities_file_path:
        config.pii_entities_file_path = entities_file_path

    global entity_loader
    entity_loader = PresidioEntityLoader(config)
    entity_loader.update_entities()

    _register_udfs(config)


init()
