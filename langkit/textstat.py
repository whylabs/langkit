from logging import getLogger
import textstat
from whylogs.core.datatypes import String
from whylogs.experimental.core.udf_schema import register_type_udf


diagnostic_logger = getLogger(__name__)

# score metrics


@register_type_udf(String, namespace="performance", schema_name="text_standard_component")
def flesch_kincaid_grade(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.flesch_kincaid_grade(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in flesch kincaid grade udf: {e}"
            )
            series_results.append(None)
    return series_results


@register_type_udf(String, namespace="performance")
def flesch_reading_ease(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.flesch_reading_ease(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in flesch reading ease udf: {e}"
            )
            series_results.append(None)
    return series_results



@register_type_udf(String, namespace="performance", schema_name="text_standard_component")
def smog_index(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.smog_index(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in smog index udf: {e}"
            )
            series_results.append(None)
    return series_results


@register_type_udf(String, namespace="performance", schema_name="text_standard_component")
def coleman_liau_index(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.coleman_liau_index(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in coleman liau udf: {e}"
            )
            series_results.append(None)
    return series_results


@register_type_udf(String, namespace="performance")
def automated_readability_index(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.automated_readability_index(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in automated readability index udf: {e}"
            )
            series_results.append(None)
    return series_results


@register_type_udf(col_type=String, namespace="performance", schema_name="text_standard_component")
def dale_chall_readability_score(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.dale_chall_readability_score(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in dale chall readability udf: {e}"
            )
            series_results.append(None)
    return series_results


@register_type_udf(String, namespace="performance", schema_name="text_standard_component")
def linsear_write_formula(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.linsear_write_formula(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in linsear write formula udf: {e}"
            )
            series_results.append(None)
    return series_results


@register_type_udf(String, namespace="performance", schema_name="text_standard_component")
def gunning_fog(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.gunning_fog(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in gunning fog udf: {e}"
            )
            series_results.append(None)
    return series_results


@register_type_udf(String, namespace="performance")
def aggregate_reading_level(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.text_standard(text, float_output=True)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in aggregate reading level udf: {e}"
            )
            series_results.append(None)
    return series_results


@register_type_udf(String, namespace="performance", schema_name="es")
def fernandez_huerta(strings) -> float:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.fernandez_huerta(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in fernandez huerta udf: {e}"
            )
            series_results.append(None)
    return series_results


@register_type_udf(String, namespace="performance", schema_name="es")
def szigriszt_pazos(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.szigriszt_pazos(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in szigriszt pazos udf: {e}"
            )
            series_results.append(None)
    return series_results


@register_type_udf(String, namespace="performance", schema_name="es")
def gutierrez_polini(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.gutierrez_polini(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in gutierrez polini udf: {e}"
            )
            series_results.append(None)
    return series_results


@register_type_udf(String, namespace="performance", schema_name="es")
def crawford(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.crawford(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in crawford udf: {e}"
            )
            series_results.append(None)
    return series_results


@register_type_udf(String, namespace="performance", schema_name="it")
def gulpease_index(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.gulpease_index(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in gulpease udf: {e}"
            )
            series_results.append(None)
    return series_results


@register_type_udf(String, namespace="performance", schema_name="ar")
def osman(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.osman(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in osman udf: {e}"
            )
            series_results.append(None)
    return series_results


# count metrics


@register_type_udf(String, namespace="performance")
def syllable_count(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.syllable_count(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in syllable count udf: {e}"
            )
            series_results.append(None)
    return series_results


@register_type_udf(String, namespace="performance")
def lexicon_count(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.lexicon_count(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in lexicon count udf: {e}"
            )
            series_results.append(None)
    return series_results


@register_type_udf(String, namespace="performance")
def sentence_count(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.sentence_count(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in sentence count udf: {e}"
            )
            series_results.append(None)
    return series_results


@register_type_udf(String, namespace="performance")
def character_count(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.char_count(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in char count udf: {e}"
            )
            series_results.append(None)
    return series_results


@register_type_udf(String, namespace="performance")
def letter_count(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.letter_count(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in letter count udf: {e}"
            )
            series_results.append(None)
    return series_results

@register_type_udf(String, namespace="performance")
def polysyllable_count(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.polysyllabcount(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in polysyllab count udf: {e}"
            )
            series_results.append(None)
    return series_results

@register_type_udf(String, namespace="performance")
def monosyllable_count(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.monosyllabcount(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in monosyllab count udf: {e}"
            )
            series_results.append(None)
    return series_results

@register_type_udf(String, namespace="performance")
def difficult_words(strings) -> list:
    series_results = []
    for text in strings:
        try:
            result = textstat.textstat.difficult_words(text)
            series_results.append(result)
        except Exception as e:
            diagnostic_logger.warn(
                f"Exception in difficult words count udf: {e}"
            )
            series_results.append(None)
    return series_results


def init():
    diagnostic_logger.info("Initialized textstat metrics.")
