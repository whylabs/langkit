from functools import partial

import pandas as pd
from textstat import textstat

from langkit.core.metric import Metric, MetricCreator, SingleMetric, SingleMetricResult, UdfInput
from langkit.metrics.text_statistics_types import TextStat


def textstat_module(stat: TextStat, column_name: str) -> Metric:
    def udf(text: pd.DataFrame) -> SingleMetricResult:
        stat_func = getattr(textstat, stat)
        metrics = [stat_func(it) for it in UdfInput(text).iter_column_rows(column_name)]
        return SingleMetricResult(metrics)

    return SingleMetric(
        name=f"{column_name}.stats.{stat}",
        input_names=[column_name],
        evaluate=udf,
    )


__reading_ease_module = partial(textstat_module, "flesch_reading_ease")
prompt_reading_ease_metric = partial(__reading_ease_module, column_name="prompt")
response_reading_ease_metric = partial(__reading_ease_module, column_name="response")
prompt_response_reading_ease_module = [prompt_reading_ease_metric, response_reading_ease_metric]


__flesch_kincaid_grade_metric = partial(textstat_module, "flesch_kincaid_grade")
prompt_grade_metric = partial(__flesch_kincaid_grade_metric, column_name="prompt")
response_grade_metric = partial(__flesch_kincaid_grade_metric, column_name="response")
prompt_response_grade_metric = [
    prompt_grade_metric,
    response_grade_metric,
]


__char_count_module = partial(textstat_module, "char_count")
prompt_char_count_metric = partial(__char_count_module, column_name="prompt")
response_char_count_metric = partial(__char_count_module, column_name="response")
prompt_response_char_count_module = [prompt_char_count_metric, response_char_count_metric]


__syllable_count_module = partial(textstat_module, "syllable_count")
prompt_syllable_count_metric = partial(__syllable_count_module, column_name="prompt")
response_syllable_count_metric = partial(__syllable_count_module, column_name="response")
prompt_response_syllable_count_module = [prompt_syllable_count_metric, response_syllable_count_metric]


__lexicon_count_module = partial(textstat_module, "lexicon_count")
prompt_lexicon_count_metric = partial(__lexicon_count_module, column_name="prompt")
response_lexicon_count_metric = partial(__lexicon_count_module, column_name="response")
prompt_response_lexicon_count_module = [prompt_lexicon_count_metric, response_lexicon_count_metric]


__sentence_count_module = partial(textstat_module, "sentence_count")
prompt_sentence_count_metric = partial(__sentence_count_module, column_name="prompt")
response_sentence_count_metric = partial(__sentence_count_module, column_name="response")
prompt_response_sentence_count_module = [prompt_sentence_count_metric, response_sentence_count_metric]


__letter_count_module = partial(textstat_module, "letter_count")
prompt_letter_count_metric = partial(__letter_count_module, column_name="prompt")
response_letter_count_metric = partial(__letter_count_module, column_name="response")
prompt_response_letter_count_module = [prompt_letter_count_metric, response_letter_count_metric]


__difficult_words_module = partial(textstat_module, "difficult_words")
prompt_difficult_words_metric = partial(__difficult_words_module, column_name="prompt")
response_difficult_words_metric = partial(__difficult_words_module, column_name="response")
prompt_response_difficult_words_module = [prompt_difficult_words_metric, response_difficult_words_metric]


prompt_response_textstat_module: MetricCreator = [
    *prompt_response_reading_ease_module,
    *prompt_response_grade_metric,
    *prompt_response_char_count_module,
    *prompt_response_syllable_count_module,
    *prompt_response_lexicon_count_module,
    *prompt_response_sentence_count_module,
    *prompt_response_letter_count_module,
    *prompt_response_difficult_words_module,
]

prompt_textstat_metric: MetricCreator = [
    prompt_reading_ease_metric,
    prompt_grade_metric,
    prompt_char_count_metric,
    prompt_syllable_count_metric,
    prompt_lexicon_count_metric,
    prompt_sentence_count_metric,
    prompt_letter_count_metric,
    prompt_difficult_words_metric,
]

response_textstat_metric: MetricCreator = [
    response_reading_ease_metric,
    response_grade_metric,
    response_char_count_metric,
    response_syllable_count_metric,
    response_lexicon_count_metric,
    response_sentence_count_metric,
    response_letter_count_metric,
    response_difficult_words_metric,
]
