from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd

from langkit.metrics.util import LazyInit


# TODO make this generic and add a filter ability to ensure that it only delivers the things
# you want instead of a bunch of Any
class UdfInput:
    """
    Utility class for iterating over the input data to a UDF row by row.
    """

    def __init__(self, text: Union[pd.DataFrame, Dict[str, List[Any]]]) -> None:
        self.text = text

    def iter_column_rows(self, column_name: str) -> Iterator[Any]:
        if column_name not in self.text:
            return iter([])

        if isinstance(self.text, pd.DataFrame):
            col = cast("pd.Series[Any]", self.text[column_name])
            return iter(col)
        else:
            return iter(self.text[column_name])

    def to_numpy(self, column_name: str) -> np.ndarray[Any, Any]:
        if column_name not in self.text:
            raise ValueError(f"Column {column_name} not found in {self.text}")

        if isinstance(self.text, pd.DataFrame):
            col = cast("pd.Series[Any]", self.text[column_name])
            return cast(np.ndarray[Any, Any], col.to_numpy())  # type: ignore[reportUnknownMemberType]
        else:
            return np.array(self.text[column_name])

    def to_list(self, column_name: str) -> List[Any]:
        if column_name not in self.text:
            raise KeyError(f"Column {column_name} not found in {self.text}")

        if isinstance(self.text, pd.DataFrame):
            col = cast("pd.Series[Any]", self.text[column_name])
            return col.to_list()

        return self.text[column_name]


MetricResultType = Union[
    Sequence[Optional[int]],
    Sequence[Optional[float]],
    Sequence[Optional[str]],
    Sequence[int],
    Sequence[float],
    Sequence[str],
]


@dataclass(frozen=True)
class SingleMetricResult:
    """
    This is the type that all of the UDFs should return.
    """

    metrics: MetricResultType


@dataclass(frozen=True)
class MultiMetricResult:
    """
    This is the type that all of the UDFs should return.
    """

    metrics: Sequence[MetricResultType]


MetricResult = Union[SingleMetricResult, MultiMetricResult]


@dataclass(frozen=True)
class SingleMetric:
    name: str  # Basically the output name
    input_names: List[str]
    evaluate: Callable[[pd.DataFrame], SingleMetricResult]
    init: Optional[Callable[[], None]] = None
    cache_assets: Optional[Callable[[], None]] = None


@dataclass(frozen=True)
class MultiMetric:
    # Splitting the metric into single/multi can be a bit verbose, but it lets us know all of the metric names
    # that are going to be generated upfront without having to evaluate all of the metrics to find out.
    names: List[str]
    input_names: List[str]
    evaluate: Callable[[pd.DataFrame], MultiMetricResult]
    init: Optional[Callable[[], None]] = None
    cache_assets: Optional[Callable[[], None]] = None


Metric = Union[SingleMetric, MultiMetric]

# Don't allow a raw Metric to be a Module because wrapping it in a callable of some kind
# lets us defer/manage side effects.
MetricCreator = Union[
    List["MetricCreator"],
    Callable[[], "MetricCreator"],
    Callable[[], List["MetricCreator"]],
    Callable[[], Metric],
    Callable[[], List[Metric]],
    List[Callable[[], Metric]],
]


@dataclass(frozen=True)
class WorkflowMetricConfig:
    metrics: List[Metric]


class MetricNameCapture:
    """
    Nice little wrapper that evaluates metric creators for you under the hood while allowing
    you get get the metric name references.
    """

    def __init__(self, creator: MetricCreator) -> None:
        self._creator = creator
        self._metrics = LazyInit(lambda: WorkflowMetricConfigBuilder().add(self._creator).build().metrics)
        self._metric_names = LazyInit(lambda: MetricNameCapture.__get_metric_names(self._metrics.value))

    @staticmethod
    def __get_metric_names(metrics: List[Metric]) -> List[str]:
        names: List[str] = []
        for metric in metrics:
            if isinstance(metric, SingleMetric):
                names.append(metric.name)
            else:
                names.extend(metric.names)
        return names

    def __call__(self) -> MetricCreator:
        return lambda: self._metrics.value

    @property
    def metric_names(self) -> List[str]:
        return self._metric_names.value


class WorkflowMetricConfigBuilder:
    def __init__(self, metric_creators: Optional[List[MetricCreator]] = None) -> None:
        super().__init__()
        self._modules: List[MetricCreator] = metric_creators or []

    def add(self, module: MetricCreator) -> "WorkflowMetricConfigBuilder":
        if isinstance(module, list):
            self._modules.extend(module)
        elif callable(module):
            self._modules.append(module)
        else:
            self._modules.append(module)

        return self

    def _build_metrics(self, modules: List[MetricCreator]) -> List[Metric]:
        schemas: List[Metric] = []
        for module in modules:
            if callable(module):
                schema = module()
                if isinstance(schema, SingleMetric) or isinstance(schema, MultiMetric):
                    schemas.append(schema)
                elif isinstance(schema, list):
                    for s in schema:
                        if isinstance(s, SingleMetric) or isinstance(s, MultiMetric):
                            schemas.append(s)
                        else:
                            schemas.extend(self._build_metrics([s]))
                else:
                    s = schema
                    schemas.extend(self._build_metrics([schema]))
            else:
                for s in module:
                    schemas.extend(self._build_metrics([s]))

        return schemas

    def build(self) -> WorkflowMetricConfig:
        schemas: List[Metric] = self._build_metrics(self._modules)

        return WorkflowMetricConfig(metrics=schemas)
