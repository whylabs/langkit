from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Mapping

import pandas as pd

from langkit.metrics.metric import (
    EvaluationConfifBuilder,
    Metric,
    MetricCreator,
    MetricResult,
    MultiMetricResult,
    SingleMetric,
    SingleMetricResult,
)
from langkit.pipeline.validation import ValidationResult, Validator


@dataclass(frozen=True)
class EvaluationResult:
    features: pd.DataFrame
    validation_results: ValidationResult

    def get_failed_ids(self) -> List[int]:
        return list(set([it.id for it in self.validation_results.report]))

    def get_failed_rows(self) -> pd.DataFrame:
        return self.features.loc[self.get_failed_ids()]


# Basically, any side effect that doesn't mutate the inputs is fine here
class Hook(ABC):
    @abstractmethod
    def post_evaluation(self, metric_results: Mapping[str, MetricResult]) -> None:
        pass

    @abstractmethod
    def post_validation(
        self, metric_results: Mapping[str, MetricResult], results: pd.DataFrame, validation_results: List[ValidationResult]
    ) -> None:
        # Can send a notification or call a webhook or log or whatever
        pass


# TODO questions
# - do we want to allow multiple outputs or do we want to restrict people to a single series as an output?
# - maybe validation and feature extraction are just separate things? Or maybe validation just has to take place on the final combined df


# TODO requirements
# - async/sync. What even makes sense here? How do you do async feature extraction? Where would you get the results?
# - metric timeout and default actions
# - handle error metrics. Probably in a similar way to metric timeouts
# - templated responses.
# - traces. Probably just a trace hook
# - DONE replace PII with <redacted>


class EvaluationWorkflow:
    def __init__(self, metrics: List[MetricCreator], hooks: List[Hook], validators: List[Validator]) -> None:
        self.metrics = EvaluationConfifBuilder().add(metrics).build()
        self.hooks = hooks
        self.validators = validators

    def _condense_metric_results(self, metric_results: Dict[str, SingleMetricResult]) -> pd.DataFrame:
        full_df = pd.DataFrame()
        for metric_name, result in metric_results.items():
            full_df[metric_name] = result.metrics

        return full_df

    def _condense_validation_results(self, validation_results: List[ValidationResult]) -> ValidationResult:
        result = ValidationResult()
        for validation_result in validation_results:
            result.report.extend(validation_result.report)
        return result

    def get_metric_names(self) -> List[str]:
        names: List[str] = []
        for metric in self.metrics.metrics:
            if isinstance(metric, SingleMetric):
                names.append(metric.name)
            else:
                names.extend(metric.names)
        return names

    def evaluate(self, df: pd.DataFrame) -> EvaluationResult:
        # Evaluation
        metric_results: Dict[str, SingleMetricResult] = {}
        for metric in self.metrics.metrics:
            if isinstance(metric, SingleMetric):
                result = metric.evaluate(df)
                self._validate_evaluate(df, metric, result)
                metric_results[metric.name] = result

                # Metric Validation
                for validator in self.validators:
                    validation_result = validator.validate_metric(metric.name, result)
                    if validation_result and validation_result.report:
                        # TODO make this only short circuit the metric, not the whole evaluation. Each row should be evaluated
                        return EvaluationResult(pd.DataFrame(), validation_result)

            else:
                # MultiMetrics are basically just converted into single metrics asap.
                result = metric.evaluate(df)
                self._validate_evaluate(df, metric, result)
                for metric_name, metric_result in zip(metric.names, result.metrics):
                    single_metric = SingleMetricResult(metric_result)
                    metric_results[metric_name] = single_metric

                    # Metric Validation
                    for validator in self.validators:
                        validation_result = validator.validate_metric(metric_name, single_metric)
                        if validation_result and validation_result.report:
                            # TODO make this only short circuit the metric, not the whole evaluation. Each row should be evaluated
                            return EvaluationResult(pd.DataFrame(), validation_result)

        # Hooks
        for action in self.hooks:
            action.post_evaluation(metric_results)

        # Validation
        condensed = self._condense_metric_results(metric_results)
        condensed["id"] = condensed.index
        full_df = condensed.copy()  # guard against mutations
        validation_results: List[ValidationResult] = []
        for validator in self.validators:
            result2 = validator.validate_result(condensed)
            if result2 and result2.report:
                validation_results.append(result2)

        # Post validation hook
        for action in self.hooks:
            action.post_validation(metric_results, full_df.copy(), validation_results)

        return EvaluationResult(full_df, self._condense_validation_results(validation_results))

    def _validate_evaluate(self, input_df: pd.DataFrame, metric: Metric, metric_result: MetricResult) -> None:
        """
        Validate the oultput of the metrics
        """

        if isinstance(metric, SingleMetric):
            assert len(input_df) == len(metric_result.metrics)
        else:
            assert len(metric.names) == len(metric_result.metrics)

        if isinstance(metric_result, MultiMetricResult):
            for result in metric_result.metrics:
                assert len(input_df) == len(result)
