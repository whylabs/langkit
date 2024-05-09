import inspect
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union, cast, overload

import pandas as pd
from typing_extensions import NotRequired, TypedDict

from langkit.core.context import Context
from langkit.core.metric import (
    Metric,
    MetricCreator,
    MetricResult,
    MultiEvaluate,
    MultiEvaluateWithContext,
    MultiMetricResult,
    SingleEvaluate,
    SingleEvaluateWithContext,
    SingleMetric,
    SingleMetricResult,
    WorkflowMetricConfigBuilder,
)
from langkit.core.validation import ValidationResult, Validator

logger = logging.getLogger(__name__)


class InputContextItem(TypedDict):
    content: str
    metadata: NotRequired[Dict[str, str]]


class InputContext(TypedDict):
    entries: List[InputContextItem]


class Row(TypedDict):
    prompt: NotRequired[str]
    response: NotRequired[str]
    context: NotRequired[InputContext]


@dataclass(frozen=True)
class RunPerf:
    init_total_sec: float
    metrics_time_sec: Dict[str, float]
    metrics_total_sec: float
    context_time_sec: Dict[str, float]
    context_total_sec: float
    validation_total_sec: float
    workflow_total_sec: float


@dataclass(frozen=True)
class WorkflowResult:
    metrics: pd.DataFrame
    validation_results: ValidationResult
    perf_info: RunPerf

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


@dataclass(frozen=True)
class MetricFilterOptions:
    by_required_inputs: Optional[List[List[str]]] = None


@dataclass(frozen=True)
class RunOptions:
    metric_filter: Optional[MetricFilterOptions] = None


class Workflow:
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
            validators: A list of validators to run after the workflow is complete.
            callbacks: A list of callbacks to run after the workflow is complete.
            lazy_init: If True, the metrics will not be initialized until the first call to run.
            cache_assets: If True, the assets required for the metrics will be cached during inititialization.
        """
        self.callbacks = callbacks or []
        self.metrics_config = WorkflowMetricConfigBuilder().add(metrics).build()
        self.validators = validators or []
        self._initialized = False
        self._cache_assets = cache_assets

        # Get the context dependencies from the metrics and dedupe them with set
        self._context_dependencies = list(
            set([dependency for sublist in self.metrics_config.metrics for dependency in (sublist.context_dependencies or [])])
        )

        if not lazy_init:
            self.init()

    def init(self) -> None:
        if self._initialized:
            return

        self._initialized = True

        for dependency in self._context_dependencies:
            if self._cache_assets:
                dependency.cache_assets()

            dependency.init()

        # TODO Maybe we should keep track of which already were initialized and only init the ones that weren't in this pipeline?
        # I prefer init just be idempotent but it might be hard for people to get right.
        metric_names: Set[str] = set()
        for metric in self.metrics_config.metrics:
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
                    f"Validator {validator} has target metric names ({targets}) but this workflow is "
                    f"only generating metrics for these: {metric_names}"
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
        for metric in self.metrics_config.metrics:
            if isinstance(metric, SingleMetric):
                names.append(metric.name)
            else:
                names.extend(metric.names)
        return names

    def get_metric_metadata(self) -> Dict[str, Dict[str, Any]]:
        metadata: Dict[str, Dict[str, Any]] = {}
        for metric in self.metrics_config.metrics:
            if metric.metadata:
                if isinstance(metric, SingleMetric):
                    metadata[metric.name] = metric.metadata
                else:
                    for name in metric.names:
                        metadata[name] = metric.metadata
        return metadata

    @overload
    def run(self, data: pd.DataFrame, options: Optional[RunOptions] = None) -> WorkflowResult:
        """
        This form is intended for batch inputs,
        where the input is a pandas DataFrame.
        """
        ...

    @overload
    def run(self, data: Row, options: Optional[RunOptions] = None) -> WorkflowResult:
        """
        This form is intended for single row inputs,
        where the input is a dictionary with the keys "prompt" and "response".
        """
        ...

    @overload
    def run(self, data: Dict[str, str], options: Optional[RunOptions] = None) -> WorkflowResult:
        """
        This form doesn't assume the "prompt" and "response" key names.
        This would be required in cases where the user wants to use different
        column names, for example "question" and "answer", or "input" and "output".
        """
        ...

    def run(self, data: Union[pd.DataFrame, Row, Dict[str, str]], options: Optional[RunOptions] = None) -> WorkflowResult:
        start = time.perf_counter()
        init_start = time.perf_counter()
        self.init()
        init_end = time.perf_counter() - init_start

        if not isinstance(data, pd.DataFrame):
            if not is_dict_input(data):
                raise ValueError("Input must be a pandas DataFrame or a dictionary with string keys and string values")
            df = pd.DataFrame([data])
        else:
            df = data

        # Setup context
        all_context_start = time.perf_counter()
        context_dependency_times: List[Tuple[str, float]] = []
        context = Context()
        for dependency in self._context_dependencies:
            context_dependency_start = time.perf_counter()
            dependency.populate_request(context, df)
            context_dependency_times.append((dependency.name(), round(time.perf_counter() - context_dependency_start, 3)))

        all_context_end = time.perf_counter() - all_context_start

        # Metrics
        metric_results: Dict[str, SingleMetricResult] = {}
        all_metrics_start = time.perf_counter()
        metric_times: List[Tuple[str, float]] = []

        if options and options.metric_filter and options.metric_filter.by_required_inputs:
            by_required_inputs_set = frozenset([frozenset(x) for x in options.metric_filter.by_required_inputs])
            metrics_to_run = [metric for metric in self.metrics_config.metrics if frozenset(metric.input_names) in by_required_inputs_set]
            if not metrics_to_run:
                raise ValueError(
                    f"No metrics to run. Filters {options.metric_filter.by_required_inputs} did "
                    f"not match any metrics {self.get_metric_names()}"
                )
        else:
            metrics_to_run = self.metrics_config.metrics

        for metric in metrics_to_run:
            # check that the dataframe has the metric.input_name present, or else skip
            if not all([input_name in df.columns for input_name in metric.input_names]):
                logger.debug(f"Skipping metric {metric} because {metric.input_names} is not present in the input dataframe")
                continue

            metric_start = time.perf_counter()
            if isinstance(metric, SingleMetric):
                param_count = len(inspect.signature(metric.evaluate).parameters)

                if param_count == 2:
                    fn = cast(SingleEvaluateWithContext, metric.evaluate)
                    result = fn(df, context)
                else:
                    fn = cast(SingleEvaluate, metric.evaluate)
                    result = fn(df)

                self._validate_evaluate(df, metric, result)
                metric_results[metric.name] = result

                metric_times.append((metric.name, round(time.perf_counter() - metric_start, 3)))

            else:
                # MultiMetrics are basically just converted into single metrics asap.
                param_count = len(inspect.signature(metric.evaluate).parameters)
                if param_count == 2:
                    fn = cast(MultiEvaluateWithContext, metric.evaluate)
                    result = fn(df, context)
                else:
                    fn = cast(MultiEvaluate, metric.evaluate)
                    result = fn(df)

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

        # TODO set column names `metric` and `value`
        full_df = condensed.copy()  # guard against mutations
        validation_results: List[ValidationResult] = []
        all_validators_start = time.perf_counter()
        for validator in self.validators:
            validator_columns = validator.get_target_metric_names()
            # make sure that all of the columns that the validator needs are present in the input dataframe
            if not set(validator_columns).issubset(condensed.columns):
                logger.debug(
                    f"Skipping validator {validator} because it requires columns {validator_columns } "
                    f"which are not present in the input dataframe"
                )
                continue

            # Only pass the series that the validator asks for to the validator. This ensrues that the target names in the validator
            # actually mean something so we can use them for valdation.
            target_subset = condensed[validator.get_target_metric_names() + ["id"]]
            result2 = validator.validate_result(target_subset)
            if result2 and result2.report:
                validation_results.append(result2)

        all_validators_end = time.perf_counter() - all_validators_start

        # Post validation hook
        for callback in self.callbacks:
            try:
                callback.post_validation(df.copy(), metric_results, full_df.copy(), validation_results)
            except Exception as e:
                logger.exception(f"Callback {callback} failed with exception {e}")

        # Performance
        run_perf = RunPerf(
            metrics_time_sec=dict(metric_times),
            workflow_total_sec=round(time.perf_counter() - start, 3),
            validation_total_sec=round(all_validators_end, 3),
            metrics_total_sec=round(all_metrics_end, 3),
            context_time_sec=dict(context_dependency_times),
            context_total_sec=round(all_context_end, 3),
            init_total_sec=round(init_end, 3),
        )

        return WorkflowResult(full_df, self._condense_validation_results(validation_results), perf_info=run_perf)

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


def is_input_context_item(variable: object) -> bool:
    if not isinstance(variable, dict):
        return False

    variable = cast(InputContextItem, variable)
    return "content" in variable and ("metadata" in variable or len(variable) == 1)


def is_input_context(variable: object) -> bool:
    if not isinstance(variable, dict):
        return False
    if "entries" not in variable:
        return False

    if not isinstance(variable["entries"], list):
        return False

    variable = cast(InputContext, variable)
    if len(variable) != 1:
        return False

    return all(is_input_context_item(value) for value in variable["entries"])


def is_dict_input(variable: object) -> bool:
    if not isinstance(variable, dict):
        return False
    # Check if all values in the dictionary are strings
    return all(isinstance(value, str) or is_input_context(value) for value in variable.values())  # type: ignore[reportUnknownMemberType]
