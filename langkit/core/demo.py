import re
from pprint import pprint
from typing import List, Mapping

import pandas as pd

from langkit.core.metric import MetricCreator, MetricNameCapture, MetricResult, MultiMetric, MultiMetricResult
from langkit.core.validation import ValidationResult, create_validator
from langkit.core.workflow import EvaluationWorkflow, Hook
from langkit.metrics.library import lib
from langkit.metrics.regexes.regex_loader import CompiledPattern, CompiledPatternGroups

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


class MyHook(Hook):
    def post_evaluation(self, metric_results: Mapping[str, MetricResult]) -> None:
        print()
        print("## POST_EVALUATION")
        # pprint(metric_results)
        print()

    def post_validation(
        self, metric_results: Mapping[str, MetricResult], results: pd.DataFrame, validation_results: List[ValidationResult]
    ) -> None:
        print()
        print("## POST_VALIDATION")
        # pprint(metric_results)
        # pprint(validation_results)
        print()


hooks: List[Hook] = [MyHook()]

patterns = CompiledPatternGroups(
    patterns=[
        CompiledPattern(name="my_pii", expressions=[re.compile("foobar")], substitutions=["<redacted>"]),
    ]
)


def my_multi_metric() -> MetricCreator:
    def udf(text: pd.DataFrame) -> MultiMetricResult:
        # for every row in text, create a list of ints
        metrics: List[int] = []
        for i, _row in text.iterrows():  # type: ignore
            metrics.append(1000)

        # 3 of them, because we have 3 names in the output
        return MultiMetricResult(metrics=[metrics, metrics, metrics])

    return lambda: MultiMetric(names=["a", "b", "c"], input_name="prompt", evaluate=udf)


# Keep a reference to this one so we can use it to filtr the output for these metrics later
prompt_text_stat = MetricNameCapture(lib.text_stat.prompt())

metrics = [
    prompt_text_stat,
    lib.text_stat.char_count.response(),
    lib.text_stat.reading_ease.response(),
    lib.text_stat.reading_ease.create("gunning_fog", "prompt"),
    lib.substitutions.prompt(file_or_patterns=patterns),
    lib.substitutions.response(file_or_patterns=patterns),
    my_multi_metric(),
]

# TODO make enum of common names for the metrics so they aren't just strings
# TODO do a custom metric
validators = [
    create_validator("prompt.char_count", upper_threshold=10),
    create_validator("prompt.flesch_reading_ease", lower_threshold=80),
    create_validator("prompt.flesch_kincaid_grade", upper_threshold=2),
    create_validator("prompt.gunning_fog", upper_threshold=2),
    create_validator("response.char_count", upper_threshold=10),
    create_validator("response.flesch_reading_ease", lower_threshold=80),
    create_validator("response.flesch_kincaid_grade", upper_threshold=2),
    create_validator("response.gunning_fog", upper_threshold=2),
]

wf = EvaluationWorkflow(metrics, hooks, validators)

if __name__ == "__main__":
    # Evaluating prompts (or whatever)
    prompts_and_responses_df = pd.DataFrame(
        {
            "prompt": ["This is a prompt", "This is another prompt", "good", "I'm going to say foobar, I'll do it."],
            "response": ["This is a response", "This is another response", "nice", "bro don't say foobar"],
        }
    )

    result = wf.evaluate(prompts_and_responses_df)

    print()
    print("## METRIC NAMES")
    pprint(wf.get_metric_names())

    # Using the outputs
    combined_df = pd.concat([prompts_and_responses_df, result.features], axis=1)  # type: ignore

    print()
    print("## FEATURES")
    print(result.features.transpose())  # type: ignore
    print()
    print("## COMBINED WITH DATA")
    print(combined_df.transpose())  # type: ignore
    print()
    print("## VALIDATION RESULTS")
    pprint(result.validation_results, width=200)

    failed_row_ids = set([it.id for it in result.validation_results.report])
    print()
    print("## FAILED ROWS")
    print(result.get_failed_ids())
    print()
    failed_rows = result.get_failed_rows()
    print(failed_rows.transpose())  # type: ignore

    # Get specific metric series. This is the metric output for each row
    print()
    print(f"## ALL PROMPT TEXT STAT METRIC NAMES: {prompt_text_stat.metric_names}")
    print(result.features[prompt_text_stat.metric_names].transpose())  # type: ignore
