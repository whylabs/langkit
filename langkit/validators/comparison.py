from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd

from langkit.core.validation import ValidationFailure, ValidationResult, Validator


def _enforce_upper_threshold(target_metric: str, upper_threshold: Union[int, float], value: Any, id: str) -> Sequence[ValidationFailure]:
    if not isinstance(value, (float, int)):
        return []

    if value > upper_threshold:
        return [
            ValidationFailure(
                id=id,
                metric=target_metric,
                details=f"Value {value} is above threshold {upper_threshold}",
                value=value,
                upper_threshold=upper_threshold,
            )
        ]

    return []


def _enforce_lower_threshold(target_metric: str, lower_threshold: Union[int, float], value: Any, id: str) -> Sequence[ValidationFailure]:
    if not isinstance(value, (float, int)):
        return []

    if value < lower_threshold:
        return [
            ValidationFailure(
                id=id,
                metric=target_metric,
                details=f"Value {value} is below threshold {lower_threshold}",
                value=value,
                lower_threshold=lower_threshold,
            )
        ]

    return []


def _enforce_upper_threshold_inclusive(
    target_metric: str, upper_threshold_inclusive: Union[int, float], value: Any, id: str
) -> Sequence[ValidationFailure]:
    if not isinstance(value, (float, int)):
        return []

    if value >= upper_threshold_inclusive:
        return [
            ValidationFailure(
                id=id,
                metric=target_metric,
                details=f"Value {value} is above or equal to threshold {upper_threshold_inclusive}",
                value=value,
                upper_threshold=upper_threshold_inclusive,
            )
        ]

    return []


def _enforce_lower_threshold_inclusive(
    target_metric: str, lower_threshold_inclusive: Union[int, float], value: Any, id: str
) -> Sequence[ValidationFailure]:
    if not isinstance(value, (float, int)):
        return []

    if value <= lower_threshold_inclusive:
        return [
            ValidationFailure(
                id=id,
                metric=target_metric,
                details=f"Value {value} is below or equal to threshold {lower_threshold_inclusive}",
                value=value,
                lower_threshold=lower_threshold_inclusive,
            )
        ]

    return []


def _enforce_one_of(target_metric: str, one_of: Set[Union[str, float, int]], value: Any, id: str) -> Sequence[ValidationFailure]:
    if value not in one_of:
        return [
            ValidationFailure(
                id=id,
                metric=target_metric,
                details=f"Value {value} is not in allowed values {one_of}",
                value=value,
                allowed_values=list(one_of),
            )
        ]
    return []


def _enforce_none_of(target_metric: str, none_of: Set[Union[str, float, int]], value: Any, id: str) -> Sequence[ValidationFailure]:
    if value in none_of:
        return [
            ValidationFailure(
                id=id,
                metric=target_metric,
                details=f"Value {value} is in disallowed values {none_of}",
                value=value,
                disallowed_values=list(none_of),
            )
        ]
    return []


def _enforce_must_be_none(target_metric: str, value: Any, id: str) -> Sequence[ValidationFailure]:
    if value is not None:
        return [
            ValidationFailure(
                id=id,
                metric=target_metric,
                details=f"Value {value} is not None",
                value=value,
                must_be_none=True,
            )
        ]
    return []


def _enforce_must_be_non_none(target_metric: str, value: Any, id: str) -> Sequence[ValidationFailure]:
    if value is None:
        return [
            ValidationFailure(
                id=id,
                metric=target_metric,
                details="Value is None",
                value=value,
                must_be_non_none=True,
            )
        ]
    return []


class ConstraintValidator(Validator):
    def __init__(
        self,
        target_metric: str,
        upper_threshold: Optional[Union[float, int]] = None,
        upper_threshold_inclusive: Optional[Union[float, int]] = None,
        lower_threshold: Optional[Union[float, int]] = None,
        lower_threshold_inclusive: Optional[Union[float, int]] = None,
        one_of: Optional[Sequence[Union[str, float, int]]] = None,
        none_of: Optional[Sequence[Union[str, float, int]]] = None,
        must_be_non_none: Optional[bool] = None,
        must_be_none: Optional[bool] = None,
    ):
        validation_functions: List[Callable[[Any, str], Sequence[ValidationFailure]]] = []

        if upper_threshold is not None:
            validation_functions.append(partial(_enforce_upper_threshold, target_metric, upper_threshold))
        if lower_threshold is not None:
            validation_functions.append(partial(_enforce_lower_threshold, target_metric, lower_threshold))
        if upper_threshold_inclusive is not None:
            validation_functions.append(partial(_enforce_upper_threshold_inclusive, target_metric, upper_threshold_inclusive))
        if lower_threshold_inclusive is not None:
            validation_functions.append(partial(_enforce_lower_threshold_inclusive, target_metric, lower_threshold_inclusive))
        if one_of is not None:
            validation_functions.append(partial(_enforce_one_of, target_metric, set(one_of)))
        if none_of is not None:
            validation_functions.append(partial(_enforce_none_of, target_metric, set(none_of)))
        if must_be_non_none is not None:
            validation_functions.append(partial(_enforce_must_be_non_none, target_metric))
        if must_be_none is not None:
            validation_functions.append(partial(_enforce_must_be_none, target_metric))

        self._target_metric = target_metric
        self._validation_functions = validation_functions

        if len(validation_functions) == 0:
            raise ValueError("At least one constraint must be provided")

    def get_target_metric_names(self) -> List[str]:
        return [self._target_metric]

    def validate_result(self, df: pd.DataFrame) -> Optional[ValidationResult]:
        failures: List[ValidationFailure] = []
        for _index, row in df.iterrows():  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            id = str(row["id"])  # pyright: ignore[reportUnknownArgumentType]
            value: Any = row[self._target_metric]
            if isinstance(value, pd.Series) and value.size == 1:
                value = value.item()
            elif isinstance(value, np.ndarray) and value.size == 1:
                value = value.item()

            for validation_function in self._validation_functions:
                failures.extend(validation_function(value, id))

        if len(failures) == 0:
            return None

        return ValidationResult(failures)
