from typing import Optional, Sequence, Union

from langkit.core.validation import Validator
from langkit.validators.comparison import ConstraintValidator


class lib:
    @staticmethod
    def constraint(
        target_metric: str,
        upper_threshold: Optional[float] = None,
        upper_threshold_inclusive: Optional[float] = None,
        lower_threshold: Optional[float] = None,
        lower_threshold_inclusive: Optional[float] = None,
        one_of: Optional[Sequence[Union[str, float, int]]] = None,
        none_of: Optional[Sequence[Union[str, float, int]]] = None,
        must_be_non_none: Optional[bool] = None,
        must_be_none: Optional[bool] = None,
    ) -> Validator:
        return ConstraintValidator(
            target_metric=target_metric,
            upper_threshold=upper_threshold,
            upper_threshold_inclusive=upper_threshold_inclusive,
            lower_threshold=lower_threshold,
            lower_threshold_inclusive=lower_threshold_inclusive,
            one_of=one_of,
            none_of=none_of,
            must_be_non_none=must_be_non_none,
            must_be_none=must_be_none,
        )
