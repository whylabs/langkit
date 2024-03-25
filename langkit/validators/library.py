from typing import List, Literal, Optional, Sequence, Union

from langkit.core.validation import Validator
from langkit.validators.comparison import (
    ConstraintValidator,
    ConstraintValidatorOptions,
    MultiColumnConstraintValidator,
    MultiColumnConstraintValidatorOptions,
)


class lib:
    class presets:
        @staticmethod
        def recommended() -> List[Validator]:
            return [
                lib.constraint(
                    target_metric="prompt.similarity.injection",
                    upper_threshold=0.5,
                ),
                lib.constraint(
                    target_metric="prompt.similarity.jailbreak",
                    upper_threshold=0.5,
                ),
                lib.constraint(
                    target_metric="response.toxicity.toxicity_score",
                    upper_threshold=0.2,
                ),
                lib.constraint(
                    target_metric="response.sentiment.sentiment_score",
                    lower_threshold=-0.2,
                ),
                lib.constraint(
                    target_metric="response.similarity.refusal",
                    lower_threshold=-0.2,
                ),
            ]

        @staticmethod
        def pii() -> List[Validator]:
            return [
                lib.constraint(
                    target_metric="prompt.pii.phone_number",
                    upper_threshold=0,
                ),
                lib.constraint(
                    target_metric="prompt.pii.email_address",
                    upper_threshold=0,
                ),
                lib.constraint(
                    target_metric="prompt.pii.us_ssn",
                    upper_threshold=0,
                ),
                lib.constraint(
                    target_metric="prompt.pii.us_bank_number",
                    upper_threshold=0,
                ),
                lib.constraint(
                    target_metric="prompt.pii.credit_card",
                    upper_threshold=0,
                ),
            ]

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
            ConstraintValidatorOptions(
                target_metric=target_metric,
                upper_threshold=upper_threshold,
                upper_threshold_inclusive=upper_threshold_inclusive,
                lower_threshold=lower_threshold,
                lower_threshold_inclusive=lower_threshold_inclusive,
                one_of=tuple(one_of) if one_of else None,
                none_of=tuple(none_of) if none_of else None,
                must_be_non_none=must_be_non_none,
                must_be_none=must_be_none,
            )
        )

    @staticmethod
    def multi_column_constraint(
        constraints: List[ConstraintValidatorOptions],
        operator: Literal["AND", "OR"] = "AND",
        report_mode: Literal["ALL_FAILED_METRICS", "FIRST_FAILED_METRIC"] = "FIRST_FAILED_METRIC",
    ) -> Validator:
        return MultiColumnConstraintValidator(
            MultiColumnConstraintValidatorOptions(constraints=tuple(constraints), operator=operator, report_mode=report_mode)
        )
