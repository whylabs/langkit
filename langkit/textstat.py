from logging import getLogger
from typing import Callable, Dict, List, Tuple, Union
import textstat
from whylogs.core.stubs import pd
from whylogs.experimental.core.udf_schema import register_dataset_udf
from . import prompt_column, response_column


diagnostic_logger = getLogger(__name__)


# score metrics

# (stat name, schema name[, udf name])
_udfs_to_register: List[Union[Tuple[str, str], Tuple[str, str, str]]] = [
    ("flesch_kincaid_grade", "text_standard_component"),
    ("flesch_reading_ease", ""),
    ("smog_index", "text_standard_component"),
    ("coleman_liau_index", "text_standard_component"),
    ("automated_readability_index", ""),
    ("dale_chall_readability_score", "text_standard_component"),
    ("linsear_write_formula", "text_standard_component"),
    ("gunning_fog", "text_standard_component"),
    ("text_standard", "", "aggregate_reading_level"),
    ("fernandez_huerta", "es"),
    ("szigriszt_pazos", "es"),
    ("gutierrez_polini", "es"),
    ("crawford", "es"),
    ("gulpease_index", "it"),
    ("osman", "ar"),
    # count metrics
    ("syllable_count", ""),
    ("lexicon_count", ""),
    ("sentence_count", ""),
    ("char_count", "", "character_count"),
    ("letter_count", ""),
    ("polysyllabcount", "", "polysyllable_count"),
    ("monosyllabcount", "", "monosyllable_count"),
    ("difficult_words", ""),
]


def wrapper(
    stat_name: str, column: str
) -> Callable[[Union[pd.DataFrame, Dict[str, List]]], Union[pd.Series, List]]:
    stat = textstat.textstat.__getattribute__(stat_name)

    def wrappee(text: Union[pd.DataFrame, Dict[str, List]]) -> Union[pd.Series, List]:
        return [stat(input) for input in text[column]]

    return wrappee


def aggregate_wrapper(
    column: str,
) -> Callable[[Union[pd.DataFrame, Dict[str, List]]], Union[pd.Series, List]]:
    stat = textstat.textstat.text_standard

    def wrappee(text: Union[pd.DataFrame, Dict[str, List]]) -> Union[pd.Series, List]:
        return [stat(input, float_output=True) for input in text[column]]

    return wrappee


def init():
    pass


init()


def _unpack(t: Union[Tuple[str, str], Tuple[str, str, str]]) -> Tuple[str, str, str]:
    return t if len(t) == 3 else (t[0], t[1], t[0])  # type: ignore


_registered = False


if not _registered:
    _registered = True
    for t in _udfs_to_register:
        stat_name, schema_name, udf = _unpack(t)
        for column in [prompt_column, response_column]:
            register_dataset_udf(
                [column], udf_name=f"{column}.{udf}", schema_name=schema_name
            )(wrapper(stat_name, column))
    for column in [prompt_column, response_column]:
        register_dataset_udf([column], udf_name=f"{column}.aggregate_reading_level")(
            aggregate_wrapper(column)
        )

    diagnostic_logger.info("Initialized textstat metrics.")
