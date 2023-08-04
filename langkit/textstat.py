from logging import getLogger
from typing import Callable, List, Tuple
import textstat
from whylogs.experimental.core.udf_schema import register_dataset_udf
from . import LangKitConfig

lang_config = LangKitConfig()


_udfs_to_register: List[Tuple[Callable, str]] = []


def langkit_udf(schema_name="") -> Callable:
    def decorator_register(func):
        global _udfs_to_register
        _udfs_to_register.append((func, schema_name))
        return func

    return decorator_register


diagnostic_logger = getLogger(__name__)

# score metrics


@langkit_udf("text_standard_component")
def flesch_kincaid_grade(text):
    result = []
    for input in text:
        result.append(textstat.textstat.flesch_kincaid_grade(input))
    return result


@langkit_udf
def flesch_reading_ease(text):
    result = []
    for input in text:
        result.append(textstat.textstat.flesch_reading_ease(input))
    return result


@langkit_udf("text_standard_component")
def smog_index(text):
    result = []
    for input in text:
        result.append(textstat.textstat.smog_index(input))
    return result


@langkit_udf("text_standard_component")
def coleman_liau_index(text):
    result = []
    for input in text:
        result.append(textstat.textstat.coleman_liau_index(input))
    return result


@langkit_udf
def automated_readability_index(text):
    result = []
    for input in text:
        result.append(textstat.textstat.automated_readability_index(input))
    return result


@langkit_udf("text_standard_component")
def dale_chall_readability_score(text):
    result = []
    for input in text:
        result.append(textstat.textstat.dale_chall_readability_score(input))
    return result


@langkit_udf("text_standard_component")
def linsear_write_formula(text):
    result = []
    for input in text:
        result.append(textstat.textstat.linsear_write_formula(input))
    return result


@langkit_udf("text_standard_component")
def gunning_fog(text):
    result = []
    for input in text:
        result.append(textstat.textstat.gunning_fog(input))
    return result


@langkit_udf
def aggregate_reading_level(text):
    result = []
    for input in text:
        result.append(textstat.textstat.text_standard(input, float_output=True))
    return result


@langkit_udf("es")
def fernandez_huerta(text):
    result = []
    for input in text:
        result.append(textstat.textstat.fernandez_huerta(input))
    return result


@langkit_udf("es")
def szigriszt_pazos(text):
    result = []
    for input in text:
        result.append(textstat.textstat.szigriszt_pazos(input))
    return result


@langkit_udf("es")
def gutierrez_polini(text):
    result = []
    for input in text:
        result.append(textstat.textstat.gutierrez_polini(input))
    return result


@langkit_udf("es")
def crawford(text):
    result = []
    for input in text:
        result.append(textstat.textstat.crawford(input))
    return result


@langkit_udf("it")
def gulpease_index(text):
    result = []
    for input in text:
        result.append(textstat.textstat.gulpease_index(input))
    return result


@langkit_udf("ar")
def osman(text):
    result = []
    for input in text:
        result.append(textstat.textstat.osman(input))
    return result


# count metrics


@langkit_udf
def syllable_count(text):
    result = []
    for input in text:
        result.append(textstat.textstat.syllable_count(input))
    return result


@langkit_udf
def lexicon_count(text):
    result = []
    for input in text:
        result.append(textstat.textstat.lexicon_count(input))
    return result


@langkit_udf
def sentence_count(text):
    result = []
    for input in text:
        result.append(textstat.textstat.sentence_count(input))
    return result


@langkit_udf
def character_count(text):
    result = []
    for input in text:
        result.append(textstat.textstat.char_count(input))
    return result


@langkit_udf
def letter_count(text):
    result = []
    for input in text:
        result.append(textstat.textstat.letter_count(input))
    return result


@langkit_udf
def polysyllable_count(text):
    result = []
    for input in text:
        result.append(textstat.textstat.polysyllabcount(input))
    return result


@langkit_udf
def monosyllable_count(text):
    result = []
    for input in text:
        result.append(textstat.textstat.monosyllabcount(input))
    return result


@langkit_udf
def difficult_words(text):
    result = []
    for input in text:
        result.append(textstat.textstat.difficult_words(input))
    return result


def init():
    diagnostic_logger.info("Initialized textstat metrics.")
    for udf, schema_name in _udfs_to_register:
        for column in [lang_config.prompt_column, lang_config.response_column]:
            register_dataset_udf(
                [column], udf_name=f"{column}.{udf.__name__}", schema_name=schema_name
            )(udf)
