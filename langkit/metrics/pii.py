from functools import partial
from typing import Any, Dict, List, Optional

import pandas as pd
import spacy
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import TransformersNlpEngine
from presidio_anonymizer import AnonymizerEngine

from langkit.core.metric import MultiMetric, MultiMetricResult


def pii_presidio_metric(
    input_name: str, language: str = "en", spacy_model: str = "en_core_web_sm", transformers_model: str = "dslim/bert-base-NER"
) -> MultiMetric:
    # Define which transformers model to use
    model_config = [
        {
            "lang_code": language,
            "model_name": {
                "spacy": spacy_model,  # use a small spaCy model for lemmas, tokens etc.
                "transformers": transformers_model,
            },
        }
    ]
    nlp_engine = TransformersNlpEngine(models=model_config)
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    anonymizer = AnonymizerEngine()

    def init() -> None:
        spacy.load(spacy_model)

    def udf(text: pd.DataFrame) -> MultiMetricResult:
        entity_types = {
            "PHONE_NUMBER": f"{input_name}.pii.phone_number",
            "EMAIL_ADDRESS": f"{input_name}.pii.email_address",
            "CREDIT_CARD": f"{input_name}.pii.credit_card",
        }

        # The order here matters
        metrics: Dict[str, List[int]] = {
            f"{input_name}.pii.phone_number": [],
            f"{input_name}.pii.email_address": [],
            f"{input_name}.pii.credit_card": [],
        }

        anonymized_metrics: Dict[str, List[Optional[str]]] = {
            f"{input_name}.pii.anonymized": [],
        }

        def process_row(
            row: pd.DataFrame,
            input_name: str,
            entity_types: Dict[str, str],
            metrics: Dict[str, List[int]],
            anonymized_metrics: Dict[str, List[Optional[str]]],
        ) -> Dict[str, List[Optional[str]]]:
            value: Any = row[input_name]
            results: List[RecognizerResult] = analyzer.analyze(text=value, language=language)

            grouped: Dict[str, int] = {"PHONE_NUMBER": 0, "EMAIL_ADDRESS": 0, "CREDIT_CARD": 0}

            for result in results:
                if result.entity_type in entity_types:
                    grouped[result.entity_type] += 1

            for entity_type, count in grouped.items():
                metrics[entity_types[entity_type]].append(count)

            anonymized_result = anonymizer.anonymize(text=value, analyzer_results=results).text if results else None  # type: ignore
            anonymized_metrics[f"{input_name}.pii.anonymized"].append(anonymized_result)

            return anonymized_metrics

        text.apply(lambda row: process_row(row, input_name, entity_types, metrics, anonymized_metrics), axis=1)  # type: ignore

        all_metrics = [
            *metrics.values(),
            *anonymized_metrics.values(),
        ]

        return MultiMetricResult(metrics=all_metrics)

    metric_names = [
        f"{input_name}.pii.phone_number",
        f"{input_name}.pii.email_address",
        f"{input_name}.pii.credit_card",
        f"{input_name}.pii.anonymized",
    ]

    return MultiMetric(names=metric_names, input_name=input_name, evaluate=udf, init=init)


prompt_presidio_pii_module = partial(pii_presidio_metric, "prompt")
response_presidio_pii_module = partial(pii_presidio_metric, "response")
prompt_response_presidio_pii_module = [prompt_presidio_pii_module, response_presidio_pii_module]
