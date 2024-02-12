from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Union

import pandas as pd


@dataclass(frozen=True)
class ValidationFailure:
    id: Union[str, int]
    metric: str
    details: str

    value: Union[int, float, str, None]
    upper_threshold: Optional[float] = None
    lower_threshold: Optional[float] = None


@dataclass(frozen=True)
class ValidationResult:
    # These are just the things that failed for logging purposes
    report: List[ValidationFailure] = field(default_factory=list)


class Validator(ABC):
    @abstractmethod
    def get_target_metric_names(self) -> List[str]:
        raise NotImplementedError()

    # TODO do we allow validation to filter things out? It would need to be exclusively row based if we did or you would end up with
    # MetricResults of varying lengths, which you really couldn't combine into a single one anymore
    # WELL, it would be ok if we just made the failures None I guess, that would preserve cardinatlity/shape
    # How do you say "remove these things because they failed?"
    # - Short circuiting the evaluation because of validation might be important, which implies validation has to occur earlier
    def validate_result(self, df: pd.DataFrame) -> Optional[ValidationResult]:
        """
        Validate the final result after all of the metrics have been evaluated.

        Args:
            df: A data frame that contains a series for every metric, as well as the original input data.
            by default, that will include a prompt and a resopnse column if both were supplied to the evaluation.
        """
        return None


def create_validator(target_metric: str, upper_threshold: Optional[float] = None, lower_threshold: Optional[float] = None) -> Validator:
    class _Validator(Validator):
        def get_target_metric_names(self) -> List[str]:
            return [target_metric]

        def validate_result(self, df: pd.DataFrame):
            failures: List[ValidationFailure] = []
            for _index, row in df.iterrows():  # type: ignore
                id = str(row["id"])  # type: ignore TODO make sure this is ok

                if upper_threshold is not None and target_metric in row and row[target_metric] > upper_threshold:
                    failures.append(
                        ValidationFailure(
                            id,
                            target_metric,
                            f"Value {row[target_metric]} is above threshold {upper_threshold}",
                            value=row[target_metric].item(),  # type: ignore TODO make sure ok
                            upper_threshold=upper_threshold,
                        )
                    )

                if lower_threshold is not None and target_metric in row and row[target_metric] < lower_threshold:
                    failures.append(
                        ValidationFailure(
                            id,
                            target_metric,
                            f"Value {row[target_metric]} is below threshold {lower_threshold}",
                            value=row[target_metric].item(),  # type: ignore TODO make sure ok
                            lower_threshold=lower_threshold,
                        )
                    )

            return ValidationResult(failures)

    return _Validator()
