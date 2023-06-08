from logging import getLogger
import textstat
from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import register_metric_udf


diagnostic_logger = getLogger(__name__)

# score metrics


@register_metric_udf(col_type=String, schema_name="text_standard_component")
def flesch_kincaid_grade(text: str) -> float:
    return textstat.textstat.flesch_kincaid_grade(text)


@register_metric_udf(col_type=String, schema_name="text_standard_component")
def flesch_reading_ease(text: str) -> float:
    return textstat.textstat.flesch_reading_ease(text)


@register_metric_udf(col_type=String, schema_name="text_standard_component")
def smog_index(text: str) -> float:
    return textstat.textstat.smog_index(text)


@register_metric_udf(col_type=String, schema_name="text_standard_component")
def coleman_liau_index(text: str) -> float:
    return textstat.textstat.coleman_liau_index(text)


@register_metric_udf(col_type=String)
def automated_readability_index(text: str) -> float:
    return textstat.textstat.automated_readability_index(text)


@register_metric_udf(col_type=String, schema_name="text_standard_component")
def dale_chall_readability_score(text: str) -> float:
    return textstat.textstat.dale_chall_readability_score(text)


@register_metric_udf(col_type=String, schema_name="text_standard_component")
def linsear_write_formula(text: str) -> float:
    return textstat.textstat.linsear_write_formula(text)


@register_metric_udf(col_type=String, schema_name="text_standard_component")
def gunning_fog(text: str) -> float:
    return textstat.textstat.gunning_fog(text)


@register_metric_udf(col_type=String)
def aggregate_reading_level(text: str) -> float:
    return textstat.textstat.text_standard(text, float_output=True)


@register_metric_udf(col_type=String, schema_name="es")
def fernandez_huerta(text: str) -> float:
    return textstat.textstat.fernandez_huerta(text)


@register_metric_udf(col_type=String, schema_name="es")
def szigriszt_pazos(text: str) -> float:
    return textstat.textstat.szigriszt_pazos(text)


@register_metric_udf(col_type=String, schema_name="es")
def gutierrez_polini(text: str) -> float:
    return textstat.textstat.gutierrez_polini(text)


@register_metric_udf(col_type=String, schema_name="es")
def crawford(text: str) -> float:
    return textstat.textstat.crawford(text)


@register_metric_udf(col_type=String, schema_name="it")
def gulpease_index(text: str) -> float:
    return textstat.textstat.gulpease_index(text)


@register_metric_udf(col_type=String, schema_name="ar")
def osman(text: str) -> float:
    return textstat.textstat.osman(text)


# count metrics


@register_metric_udf(col_type=String)
def syllable_count(text: str) -> float:
    return textstat.textstat.syllable_count(text)


@register_metric_udf(col_type=String)
def lexicon_count(text: str) -> float:
    return textstat.textstat.lexicon_count(text)


@register_metric_udf(col_type=String)
def sentence_count(text: str) -> float:
    return textstat.textstat.sentence_count(text)


@register_metric_udf(col_type=String)
def character_count(text: str) -> float:
    return textstat.textstat.char_count(text)


@register_metric_udf(col_type=String)
def letter_count(text: str) -> float:
    return textstat.textstat.letter_count(text)


@register_metric_udf(col_type=String)
def polysyllable_count(text: str) -> float:
    return textstat.textstat.polysyllabcount(text)


@register_metric_udf(col_type=String)
def monosyllable_count(text: str) -> float:
    return textstat.textstat.monosyllabcount(text)


@register_metric_udf(col_type=String)
def difficult_words(text: str) -> float:
    return textstat.textstat.difficult_words(text)


def init():
    diagnostic_logger.info("Initialized textstat metrics.")
