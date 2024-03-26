from functools import lru_cache, partial
from typing import Any, Dict, List, Optional

import pandas as pd
import spacy
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from spacy.cli.download import download  # pyright: ignore[reportUnknownVariableType]

from langkit.core.metric import MetricCreator, MultiMetric, MultiMetricResult

__default_entities = ["PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "US_SSN", "US_BANK_NUMBER"]


def __create_pii_metric_name(input_name: str, entity: str) -> str:
    return f"{input_name}.pii.{entity.lower()}"


_spacy_name = "en_core_web_lg"


@lru_cache
def _get_analyzer(lang_code: str) -> AnalyzerEngine:
    return AnalyzerEngine()


@lru_cache
def _get_anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()


@lru_cache
def _load_spacy_model():
    spacy.load(_spacy_name)


def pii_presidio_metric(
    input_name: str = "prompt",
    language: str = "en",
    entities: Optional[List[str]] = None,
) -> MultiMetric:
    if entities is None:
        entities = __default_entities.copy()

    def cache_assets() -> None:
        try:
            # Spacy isn't smart enough to avoid a bunch of extra work, it will shell out to pip
            # each time this is called.
            _load_spacy_model()
        except Exception:
            download(_spacy_name)

    def init():
        _get_analyzer(language)
        _get_anonymizer()
        _load_spacy_model()

    entity_types = {entity: __create_pii_metric_name(input_name, entity) for entity in entities}
    redacted_metric_name = f"{input_name}.pii.redacted"

    def udf(text: pd.DataFrame) -> MultiMetricResult:
        analyzer = _get_analyzer(language)
        anonymizer = _get_anonymizer()
        anonymized_metrics: Dict[str, List[Optional[str]]] = {
            redacted_metric_name: [],
        }

        metrics: Dict[str, List[int]] = {metric_name: [] for metric_name in entity_types.values()}

        def process_row(row: pd.DataFrame) -> Dict[str, List[Optional[str]]]:
            value: Any = row[input_name]
            results: List[RecognizerResult] = analyzer.analyze(text=value, language=language, entities=entities)

            grouped: Dict[str, int] = {key: 0 for key in entity_types.values()}

            for result in results:
                if result.entity_type in entity_types:
                    grouped[entity_types[result.entity_type]] += 1

            for metric_name, count in grouped.items():
                metrics[metric_name].append(count)

            anonymized_result = anonymizer.anonymize(text=value, analyzer_results=results).text if results else None  # type: ignore
            anonymized_metrics[redacted_metric_name].append(anonymized_result)
            return anonymized_metrics

        text.apply(process_row, axis=1)  # pyright: ignore[reportUnknownMemberType]

        all_metrics = [
            *metrics.values(),
            *anonymized_metrics.values(),
        ]

        return MultiMetricResult(metrics=all_metrics)

    metric_names = list(entity_types.values()) + [redacted_metric_name]
    return MultiMetric(names=metric_names, input_names=[input_name], evaluate=udf, cache_assets=cache_assets, init=init)


prompt_presidio_pii_metric = partial(pii_presidio_metric, "prompt")
response_presidio_pii_metric = partial(pii_presidio_metric, "response")
prompt_response_presidio_pii_metric: List[MetricCreator] = [prompt_presidio_pii_metric, response_presidio_pii_metric]
