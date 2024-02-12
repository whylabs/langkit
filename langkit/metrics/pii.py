from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import pandas as pd
import spacy
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import TransformersNlpEngine
from presidio_anonymizer import AnonymizerEngine

from langkit.core.metric import MetricCreator, MultiMetric, MultiMetricResult
from langkit.metrics.util import DynamicLazyInit, LazyInit


@dataclass(frozen=True)
class PresidioConfig:
    lang_code: str
    spacy: str
    transformers: str

    def to_dict(self) -> List[Dict[str, Any]]:
        return [
            {
                "lang_code": self.lang_code,
                "model_name": {
                    "spacy": self.spacy,
                    "transformers": self.transformers,
                },
            }
        ]


__default_entities = ["PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "IP_ADDRESS"]
__analyzer = DynamicLazyInit[PresidioConfig, AnalyzerEngine](
    lambda config: AnalyzerEngine(nlp_engine=TransformersNlpEngine(models=config.to_dict()))
)
__anonymizer = LazyInit(lambda: AnonymizerEngine())


def __create_pii_metric_name(input_name: str, entity: str) -> str:
    return f"{input_name}.pii.{entity.lower()}"


def pii_presidio_metric(
    input_name: str,
    language: str = "en",
    spacy_model: str = "en_core_web_sm",
    transformers_model: str = "dslim/bert-base-NER",
    entities: Optional[List[str]] = None,
) -> MultiMetric:
    if entities is None:
        entities = __default_entities.copy()

    analyzer = __analyzer.value(PresidioConfig(language, spacy_model, transformers_model))
    anonymizer = __anonymizer.value

    def cache_assets() -> None:
        spacy.load(spacy_model)

    def init() -> None:
        spacy.load(spacy_model)

    entity_types = {entity: __create_pii_metric_name(input_name, entity) for entity in entities}
    redacted_metric_name = f"{input_name}.pii.redacted"

    def udf(text: pd.DataFrame) -> MultiMetricResult:
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
    return MultiMetric(names=metric_names, input_name=input_name, evaluate=udf, init=init, cache_assets=cache_assets)


prompt_presidio_pii_metric = partial(pii_presidio_metric, "prompt")
response_presidio_pii_metric = partial(pii_presidio_metric, "response")
prompt_response_presidio_pii_metric: List[MetricCreator] = [prompt_presidio_pii_metric, response_presidio_pii_metric]
