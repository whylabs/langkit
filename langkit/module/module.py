from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Union, cast

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
            col = cast(pd.Series, self.text[column_name])
            return cast(Iterator[Any], iter(col))
        else:
            return iter(self.text[column_name])

    def to_numpy(self, column_name: str) -> np.ndarray[Any, Any]:
        if column_name not in self.text:
            raise ValueError(f"Column {column_name} not found in {self.text}")

        if isinstance(self.text, pd.DataFrame):
            col = cast(pd.Series, self.text[column_name])
            return cast(np.ndarray[Any, Any], col.to_numpy())  # type: ignore[reportUnknownMemberType]
        else:
            return np.array(self.text[column_name])

    def to_list(self, column_name: str) -> List[Any]:
        if column_name not in self.text:
            raise ValueError(f"Column {column_name} not found in {self.text}")

        if isinstance(self.text, pd.DataFrame):
            col = cast(pd.Series, self.text[column_name])
            return cast(List[Any], col.to_list())

        return self.text[column_name]


@dataclass(frozen=True)
class EvaluationResult:
    """
    This is the type that all of the UDFs should return.
    """

    metrics: Union[List[Optional[int]], List[Optional[float]], List[Optional[str]], List[int], List[float], List[str]]


# This is a UDF
EvaluateFn = Callable[[pd.DataFrame], EvaluationResult]


@dataclass(frozen=True)
class MetricConf:
    name: str  # Basically the output name
    input_name: str
    evaluate: EvaluateFn


# Don't allow a raw UdfSchemaArgs to be a Module because wrapping it in a callable of some kind
# lets us defer/manage side effects.
Module = Union[
    List["Module"],
    Callable[[], "Module"],
    Callable[[], List["Module"]],
    Callable[[], MetricConf],
    Callable[[], List[MetricConf]],
    List[Callable[[], MetricConf]],
]


@dataclass(frozen=True)
class EvaluationConfig:
    configs: List[MetricConf]


class EvaluationConfifBuilder:
    def __init__(self) -> None:
        super().__init__()
        self._modules: List[Module] = []

    def add(self, module: Module) -> "EvaluationConfifBuilder":
        if isinstance(module, list):
            self._modules.extend(module)
        elif callable(module):
            self._modules.append(module)
        else:
            self._modules.extend(module)

        return self

    def _evaluate_modules(self, modules: List[Module]) -> List[MetricConf]:
        schemas: List[MetricConf] = []
        for module in modules:
            if callable(module):
                schema = module()
                if isinstance(schema, MetricConf):
                    schemas.append(schema)
                elif isinstance(schema, list):
                    for s in schema:
                        if isinstance(s, MetricConf):
                            schemas.append(s)
                        else:
                            schemas.extend(self._evaluate_modules([s]))
                else:
                    s = schema
                    schemas.extend(self._evaluate_modules([schema]))
            else:
                # schemas.extend([m.create() for m in module])
                for m in module:
                    if callable(m):
                        schema = m()
                        if isinstance(schema, MetricConf):
                            schemas.append(schema)
                        elif isinstance(schema, list):
                            for s in schema:
                                if isinstance(s, MetricConf):
                                    schemas.append(s)
                                else:
                                    schemas.extend(self._evaluate_modules([s]))
                    else:
                        schemas.extend(self._evaluate_modules([m]))

        return schemas

    def build(self) -> EvaluationConfig:
        schemas: List[MetricConf] = self._evaluate_modules(self._modules)

        return EvaluationConfig(configs=schemas)
