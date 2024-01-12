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


@dataclass(frozen=True)
class MetricResult:
    """
    This is the type that all of the UDFs should return.
    """

    # TODO should this just be a series/dataframe?
    metrics: Union[
        Sequence[Optional[int]], Sequence[Optional[float]], Sequence[Optional[str]], Sequence[int], Sequence[float], Sequence[str]
    ]


# This is a UDF
EvaluateFn = Callable[[pd.DataFrame], MetricResult]


@dataclass(frozen=True)
# TODO maybe make this a generic of the literal name? Then we can do nice stuff in the pipeline to help actions
class Metric:
    name: str  # Basically the output name
    input_name: str
    evaluate: EvaluateFn


# Don't allow a raw UdfSchemaArgs to be a Module because wrapping it in a callable of some kind
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
    def __init__(self) -> None:
        super().__init__()
        self._modules: List[MetricCreator] = []

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
