from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Set, TypedDict, Union, overload

import pandas as pd

from langkit.core.metric import (
    EvaluationConfifBuilder,
    Metric,
    MetricCreator,
    MetricResult,
    MultiMetricResult,
    SingleMetric,
    SingleMetricResult,
)
from langkit.core.validation import ValidationResult, Validator
from langkit.metrics.util import is_dict_with_strings


class Row(TypedDict):
    prompt: str
    response: str

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
    def __init__(
        self,
        metrics: List[MetricCreator],
        hooks: Optional[List[Hook]] = None,
        validators: Optional[List[Validator]] = None,
        lazy_init=False,
    ) -> None:
        self.metrics = EvaluationConfifBuilder().add(metrics).build()
        self.hooks = hooks or []
        self.validators = validators or []
        self._initialized = False

        if not lazy_init:
            self.init()

    def init(self) -> None:
        if self._initialized:
            return

        self._initialized = True
        # TODO Maybe we should keep track of which already were initialized and only init the ones that weren't in this pipeline?
        # I prefer init just be idempotent but it might be hard for people to get right.
        metric_names: Set[str] = set()
        for metric in self.metrics.metrics:
            if metric.init:
                metric.init()

            if isinstance(metric, SingleMetric):
                metric_names.add(metric.name)
            else:
                metric_names.update(metric.names)

        for validator in self.validators:
            targets = validator.get_target_metric_names()
            if not set(targets).issubset(metric_names):
                raise ValueError(
                    f"Validator {validator} has target metric names ({targets}) that are not in the list of metrics: {metric_names}"
                )

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

    @overload
    def evaluate(self, data: pd.DataFrame) -> EvaluationResult:
        """
        This form is intended for batch evaluation,
        where the input is a pandas DataFrame.
        """
        ...

    @overload
    def evaluate(self, data: Row) -> EvaluationResult:
        """
        This form is intended for single row evaluation,
        where the input is a dictionary with the keys "prompt" and "response".
        """
        ...

    @overload
    def evaluate(self, data: Dict[str, str]) -> EvaluationResult:
        """
        This form doesn't assume the "prompt" and "response" key names.
        This would be required in cases where the user wants to use different
        column names, for example "question" and "answer", or "input" and "output".
        """
        ...

    def evaluate(self, data: Union[pd.DataFrame, Row, Dict[str, str]]) -> EvaluationResult:
        if not self._initialized:
            self.init()
        if not isinstance(data, pd.DataFrame):
            if not is_dict_with_strings(data):
                raise ValueError("Input must be a pandas DataFrame or a dictionary with string keys and string values")
            df = pd.DataFrame(data, index=[0])
        else:
            df = data
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
            # Only pass the series that the validator asks for to the validator. This ensrues that the target names in the validator
            # actually mean something so we can use them for valdation.
            target_subset = condensed[validator.get_target_metric_names() + ["id"]]
            result2 = validator.validate_result(target_subset)
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
