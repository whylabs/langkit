import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Set, Tuple, TypedDict, Union, overload

import pandas as pd

from langkit.core.metric import (
    EvaluationConfigBuilder,
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
class RunPerf:
    metrics: Dict[str, float]
    workflow_total: float
    metrics_total: float
    validation_total: float


@dataclass(frozen=True)
class EvaluationResult:
    metrics: pd.DataFrame
    validation_results: ValidationResult
    perf: RunPerf

    def get_failed_ids(self) -> List[str]:
        return list(set([it.id for it in self.validation_results.report]))

    def get_failed_rows(self) -> pd.DataFrame:
        return self.metrics.loc[self.get_failed_ids()]


# Basically, any side effect that doesn't mutate the inputs is fine here
class Callback(ABC):
    @abstractmethod
    def post_validation(
        self,
        df: pd.DataFrame,
        metric_results: Mapping[str, MetricResult],
        results: pd.DataFrame,
        validation_results: List[ValidationResult],
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
        callbacks: Optional[List[Callback]] = None,
        validators: Optional[List[Validator]] = None,
        lazy_init=False,
        cache_assets=True,
    ) -> None:
        """
        Args:
            metrics: A list of metrics to evaluate.
            validators: A list of validators to run after the evaluation is complete.
            callbacks: A list of callbacks to run after the evaluation is complete.
            lazy_init: If True, the metrics will not be initialized until the first call to run.
            cache_assets: If True, the assets required for the metrics will be cached during inititialization.
        """
        self.hooks = callbacks or []
        self.metrics = EvaluationConfigBuilder().add(metrics).build()
        self.validators = validators or []
        self._initialized = False
        self._cache_assets = cache_assets

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
            if self._cache_assets and metric.cache_assets:
                metric.cache_assets()

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
    def run(self, data: pd.DataFrame) -> EvaluationResult:
        """
        This form is intended for batch evaluation,
        where the input is a pandas DataFrame.
        """
        ...

    @overload
    def run(self, data: Row) -> EvaluationResult:
        """
        This form is intended for single row evaluation,
        where the input is a dictionary with the keys "prompt" and "response".
        """
        ...

    @overload
    def run(self, data: Dict[str, str]) -> EvaluationResult:
        """
        This form doesn't assume the "prompt" and "response" key names.
        This would be required in cases where the user wants to use different
        column names, for example "question" and "answer", or "input" and "output".
        """
        ...

    def run(self, data: Union[pd.DataFrame, Row, Dict[str, str]]) -> EvaluationResult:
        start = time.perf_counter()

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
        all_metrics_start = time.perf_counter()
        metric_times: List[Tuple[str, float]] = []

        for metric in self.metrics.metrics:
            metric_start = time.perf_counter()
            if isinstance(metric, SingleMetric):
                result = metric.evaluate(df)
                self._validate_evaluate(df, metric, result)
                metric_results[metric.name] = result

                metric_times.append((metric.name, round(time.perf_counter() - metric_start, 3)))

            else:
                # MultiMetrics are basically just converted into single metrics asap.
                result = metric.evaluate(df)
                self._validate_evaluate(df, metric, result)
                for metric_name, metric_result in zip(metric.names, result.metrics):
                    single_metric = SingleMetricResult(metric_result)
                    metric_results[metric_name] = single_metric

                metric_times.append((",".join(metric.names), round(time.perf_counter() - metric_start, 3)))

        all_metrics_end = time.perf_counter() - all_metrics_start

        # Validation
        condensed = self._condense_metric_results(metric_results)
        if "id" not in df.columns:
            condensed["id"] = df.index.astype(str)
        else:
            condensed["id"] = df["id"]

        full_df = condensed.copy()  # guard against mutations
        validation_results: List[ValidationResult] = []
        all_validators_start = time.perf_counter()
        for validator in self.validators:
            # Only pass the series that the validator asks for to the validator. This ensrues that the target names in the validator
            # actually mean something so we can use them for valdation.
            target_subset = condensed[validator.get_target_metric_names() + ["id"]]
            result2 = validator.validate_result(target_subset)
            if result2 and result2.report:
                validation_results.append(result2)

        all_validators_end = time.perf_counter() - all_validators_start

        # Post validation hook
        for action in self.hooks:
            action.post_validation(df.copy(), metric_results, full_df.copy(), validation_results)

        # Performance
        run_perf = RunPerf(
            metrics=dict(metric_times),
            workflow_total=round(time.perf_counter() - start, 3),
            validation_total=round(all_validators_end, 3),
            metrics_total=round(all_metrics_end, 3),
        )

        return EvaluationResult(full_df, self._condense_validation_results(validation_results), perf=run_perf)

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