from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Callable, List, Literal, Optional, Sequence, Set, Tuple, Union

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


@dataclass(frozen=True)
class ConstraintValidatorOptions:
    target_metric: str
    upper_threshold: Optional[Union[float, int]] = None
    upper_threshold_inclusive: Optional[Union[float, int]] = None
    lower_threshold: Optional[Union[float, int]] = None
    lower_threshold_inclusive: Optional[Union[float, int]] = None
    one_of: Optional[Tuple[Union[str, float, int], ...]] = None
    none_of: Optional[Tuple[Union[str, float, int], ...]] = None
    must_be_non_none: Optional[bool] = None
    must_be_none: Optional[bool] = None


class ConstraintValidator(Validator):
    def __init__(self, options: ConstraintValidatorOptions):
        validation_functions: List[Callable[[Any, str], Sequence[ValidationFailure]]] = []

        if options.upper_threshold is not None:
            validation_functions.append(partial(_enforce_upper_threshold, options.target_metric, options.upper_threshold))
        if options.lower_threshold is not None:
            validation_functions.append(partial(_enforce_lower_threshold, options.target_metric, options.lower_threshold))
        if options.upper_threshold_inclusive is not None:
            validation_functions.append(
                partial(_enforce_upper_threshold_inclusive, options.target_metric, options.upper_threshold_inclusive)
            )
        if options.lower_threshold_inclusive is not None:
            validation_functions.append(
                partial(_enforce_lower_threshold_inclusive, options.target_metric, options.lower_threshold_inclusive)
            )
        if options.one_of is not None:
            validation_functions.append(partial(_enforce_one_of, options.target_metric, set(options.one_of)))
        if options.none_of is not None:
            validation_functions.append(partial(_enforce_none_of, options.target_metric, set(options.none_of)))
        if options.must_be_non_none is not None:
            validation_functions.append(partial(_enforce_must_be_non_none, options.target_metric))
        if options.must_be_none is not None:
            validation_functions.append(partial(_enforce_must_be_none, options.target_metric))

        self._target_metric = options.target_metric
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


@dataclass(frozen=True)
class MultiColumnConstraintValidatorOptions:
    constraints: Tuple[ConstraintValidatorOptions, ...]
    operator: Literal["AND", "OR"] = "AND"
    report_mode: Literal["ALL_FAILED_METRICS", "FIRST_FAILED_METRIC"] = "FIRST_FAILED_METRIC"


class MultiColumnConstraintValidator(Validator):
    def __init__(
        self,
        options: MultiColumnConstraintValidatorOptions,
    ):
        """

        :param constraints: List of constraint options to validate
        :param operator: Operator to combine the constraints. Either "AND" or "OR". AND requires that all of the
            constraints trigger, while OR requires that at least one triggers.
        :param report_mode: How to report the validation result. If "FIRST_FAILED_METRIC", then this validator will
            return a single validation result when there are failures, and that validation result will contain the
            first failed metric. If "ALL_FAILED_METRICS", then this validator will return each validation failure.
        """
        self._operator = options.operator
        self._constraints = [ConstraintValidator(constraint) for constraint in options.constraints]
        self._report_mode = options.report_mode

    def get_target_metric_names(self) -> List[str]:
        target_metrics: List[str] = []
        for constraint in self._constraints:
            target_metrics.extend(constraint.get_target_metric_names())
        return target_metrics

    def validate_result(self, df: pd.DataFrame) -> Optional[ValidationResult]:
        """
        Validate all of the contraint validators and combine them using the specified operator.
        If the output of the operator is True, then return the validation result according to the report mode.
        """
        all_failures: List[ValidationFailure] = []
        for constraint in self._constraints:
            result = constraint.validate_result(df)
            if result:
                all_failures.extend(result.report)

        if len(all_failures) == 0:
            return None

        # Determine if the validation triggers by applying the operator to the list of failures
        if self._operator == "AND":
            triggered = len(all_failures) == len(self._constraints)
        else:
            triggered = len(all_failures) > 0

        if not triggered:
            return None

        if self._report_mode == "FIRST_FAILED_METRIC":
            # Create a new message that explains the failure happened because of the operator+ the names of the other failed metrics
            failure = all_failures[0]
            failure_metric_names = [failure.metric for failure in all_failures]
            trigger_details = (
                f". Triggered because of failures in {', '.join(failure_metric_names)} ({self._operator})." if len(all_failures) > 1 else ""
            )
            failure_details = f"{failure.details}{trigger_details}"
            return ValidationResult([replace(failure, details=failure_details)])

        return ValidationResult(all_failures)
