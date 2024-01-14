from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd


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
            raise ValueError(f"Column {column_name} not found in {self.text}")

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
class MetricResult:
    """
    This is the type that all of the UDFs should return.
    """

    # TODO with the ability to generate multiple metrics and metric names, we lose the ability to know metric names until actual
    # metrics are evaluated against real data.
    metrics: Union[MetricResultType, Dict[str, MetricResultType]]
    """
    The metrics that are returned by the UDF. This can be a list of values or a dict of values.
    A dict would be used to generate multiple metrics at once. The name in the dict would be appended
    to the metric name to create the full metric name, separrated by a dot.
    """


# This is a UDF
EvaluateFn = Callable[[pd.DataFrame], MetricResult]


@dataclass(frozen=True)
class Metric:
    name: str  # Basically the output name
    input_name: str
    evaluate: EvaluateFn

    def __str__(self) -> str:
        return self.name


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
class EvaluationConfig:
    metrics: List[Metric]


class EvaluationConfifBuilder:
    def __init__(self, metric_creators: Optional[List[MetricCreator]] = None) -> None:
        super().__init__()
        self._modules: List[MetricCreator] = metric_creators or []

    def add(self, module: MetricCreator) -> "EvaluationConfifBuilder":
        if isinstance(module, list):
            self._modules.extend(module)
        elif callable(module):
            self._modules.append(module)
        else:
            self._modules.extend(module)

        return self

    def _build_metrics(self, modules: List[MetricCreator]) -> List[Metric]:
        schemas: List[Metric] = []
        for module in modules:
            if callable(module):
                schema = module()
                if isinstance(schema, Metric):
                    schemas.append(schema)
                elif isinstance(schema, list):
                    for s in schema:
                        if isinstance(s, Metric):
                            schemas.append(s)
                        else:
                            schemas.extend(self._build_metrics([s]))
                else:
                    s = schema
                    schemas.extend(self._build_metrics([schema]))
            else:
                for m in module:
                    if callable(m):
                        schema = m()
                        if isinstance(schema, Metric):
                            schemas.append(schema)
                        elif isinstance(schema, list):
                            for s in schema:
                                if isinstance(s, Metric):
                                    schemas.append(s)
                                else:
                                    schemas.extend(self._build_metrics([s]))
                    else:
                        schemas.extend(self._build_metrics([m]))

        return schemas

    def build(self) -> EvaluationConfig:
        schemas: List[Metric] = self._build_metrics(self._modules)

        return EvaluationConfig(metrics=schemas)
