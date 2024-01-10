from functools import partial
from typing import Literal

import pandas as pd
from textstat import textstat

from langkit.metrics.metric import EvaluationResult, Metric, MetricConfig, UdfInput

TextStat = Literal[
    "flesch_kincaid_grade",
    "flesch_reading_ease",
    "smog_index",
    "coleman_liau_index",
    "automated_readability_index",
    "dale_chall_readability_score",
    "linsear_write_formula",
    "gunning_fog",
    "text_standard",
    "fernandez_huerta",
    "szigriszt_pazos",
    "gutierrez_polini",
    "crawford",
    "gulpease_index",
    "osman",
    "syllable_count",
    "lexicon_count",
    "sentence_count",
    "char_count",
    "letter_count",
    "polysyllabcount",
    "monosyllabcount",
    "difficult_words",
]


def __textstat_module(stat: TextStat, column_name: str) -> MetricConfig:
    def udf(text: pd.DataFrame) -> EvaluationResult:
        stat_func = getattr(textstat, stat)
        metrics = [stat_func(it) for it in UdfInput(text).iter_column_rows(column_name)]
        return EvaluationResult(metrics)

    return MetricConfig(
        name=f"{column_name}.{stat}",
        input_name=column_name,
        evaluate=udf,
    )


__reading_ease_module = partial(__textstat_module, "flesch_reading_ease")
prompt_reading_ease_module = partial(__reading_ease_module, column_name="prompt")
response_reading_ease_module = partial(__reading_ease_module, column_name="response")
prompt_response_reading_ease_module = [prompt_reading_ease_module, response_reading_ease_module]


__flesch_kincaid_grade_level_module = partial(__textstat_module, "flesch_kincaid_grade")
prompt_flesch_kincaid_grade_level_module = partial(__flesch_kincaid_grade_level_module, column_name="prompt")
response_flesch_kincaid_grade_level_module = partial(__flesch_kincaid_grade_level_module, column_name="response")
prompt_response_flesch_kincaid_grade_level_module = [
    prompt_flesch_kincaid_grade_level_module,
    response_flesch_kincaid_grade_level_module,
]


__char_count_module = partial(__textstat_module, "char_count")
prompt_char_count_module = partial(__char_count_module, column_name="prompt")
response_char_count_module = partial(__char_count_module, column_name="response")
prompt_response_char_count_module = [prompt_char_count_module, response_char_count_module]


__syllable_count_module = partial(__textstat_module, "syllable_count")
prompt_syllable_count_module = partial(__syllable_count_module, column_name="prompt")
response_syllable_count_module = partial(__syllable_count_module, column_name="response")
prompt_response_syllable_count_module = [prompt_syllable_count_module, response_syllable_count_module]


__lexicon_count_module = partial(__textstat_module, "lexicon_count")
prompt_lexicon_count_module = partial(__lexicon_count_module, column_name="prompt")
response_lexicon_count_module = partial(__lexicon_count_module, column_name="response")
prompt_response_lexicon_count_module = [prompt_lexicon_count_module, response_lexicon_count_module]


__sentence_count_module = partial(__textstat_module, "sentence_count")
prompt_sentence_count_module = partial(__sentence_count_module, column_name="prompt")
response_sentence_count_module = partial(__sentence_count_module, column_name="response")
prompt_response_sentence_count_module = [prompt_sentence_count_module, response_sentence_count_module]


__letter_count_module = partial(__textstat_module, "letter_count")
prompt_letter_count_module = partial(__letter_count_module, column_name="prompt")
response_letter_count_module = partial(__letter_count_module, column_name="response")
prompt_response_letter_count_module = [prompt_letter_count_module, response_letter_count_module]


__polysyllabcount_module = partial(__textstat_module, "polysyllabcount")
prompt_polysyllabcount_module = partial(__polysyllabcount_module, column_name="prompt")
response_polysyllabcount_module = partial(__polysyllabcount_module, column_name="response")
prompt_response_polysyllabcount_module = [prompt_polysyllabcount_module, response_polysyllabcount_module]


__monosyllabcount_module = partial(__textstat_module, "monosyllabcount")
prompt_monosyllabcount_module = partial(__monosyllabcount_module, column_name="prompt")
response_monosyllabcount_module = partial(__monosyllabcount_module, column_name="response")
prompt_response_monosyllabcount_module = [prompt_monosyllabcount_module, response_monosyllabcount_module]


__difficult_words_module = partial(__textstat_module, "difficult_words")
prompt_difficult_words_module = partial(__difficult_words_module, column_name="prompt")
response_difficult_words_module = partial(__difficult_words_module, column_name="response")
prompt_response_difficult_words_module = [prompt_difficult_words_module, response_difficult_words_module]


prompt_response_textstat_module: Metric = [
    *prompt_response_reading_ease_module,
    *prompt_response_flesch_kincaid_grade_level_module,
    *prompt_response_char_count_module,
    *prompt_response_syllable_count_module,
    *prompt_response_lexicon_count_module,
    *prompt_response_sentence_count_module,
    *prompt_response_letter_count_module,
    *prompt_response_polysyllabcount_module,
    *prompt_response_monosyllabcount_module,
    *prompt_response_difficult_words_module,
]

prompt_textstat_module: Metric = [
    prompt_reading_ease_module,
    prompt_flesch_kincaid_grade_level_module,
    prompt_char_count_module,
    prompt_syllable_count_module,
    prompt_lexicon_count_module,
    prompt_sentence_count_module,
    prompt_letter_count_module,
    prompt_polysyllabcount_module,
    prompt_monosyllabcount_module,
    prompt_difficult_words_module,
]

response_textstat_module: Metric = [
    response_reading_ease_module,
    response_flesch_kincaid_grade_level_module,
    response_char_count_module,
    response_syllable_count_module,
    response_lexicon_count_module,
    response_sentence_count_module,
    response_letter_count_module,
    response_polysyllabcount_module,
    response_monosyllabcount_module,
    response_difficult_words_module,
]
