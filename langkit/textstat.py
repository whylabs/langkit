from logging import getLogger
from typing import Callable, Dict, List, Tuple, Union
import textstat
from whylogs.core.stubs import pd
from whylogs.experimental.core.udf_schema import register_dataset_udf
from . import LangKitConfig

lang_config = LangKitConfig()

diagnostic_logger = getLogger(__name__)


# score metrics

# (stat name, schema name)
_udfs_to_register: List[Tuple[str, str]] = [
    ("flesch_kincaid_grade", "text_standard_component"),
    ("flesch_reading_ease", ""),
    ("smog_index", "text_standard_component"),
    ("coleman_liau_index", "text_standard_component"),
    ("automated_readability_index", ""),
    ("dale_chall_readability_score", "text_standard_component"),
    ("linsear_write_formula", "text_standard_component"),
    ("gunning_fog", "text_standard_component"),
    ("text_standard", ""),
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
    ("char_count", ""),
    ("letter_count", ""),
    ("polysyllabcount", ""),
    ("monosyllabcount", ""),
    ("difficult_words", ""),
]


def wrapper(
    stat_name: str,
) -> Callable[[Union[pd.DataFrame, Dict[str, List]]], Union[pd.Series, List]]:
    stat = textstat.textstat.__getattribute__(stat_name)

    def wrappee(text: Union[pd.DataFrame, Dict[str, List]]) -> Union[pd.Series, List]:
        result = []
        index = (
            text.columns[0] if isinstance(text, pd.DataFrame) else list(text.keys())[0]
        )
        for input in text[index]:
            result.append(stat(input))
        return result

    return wrappee


def init():
    diagnostic_logger.info("Initialized textstat metrics.")
    for udf, schema_name in _udfs_to_register:
        for column in [lang_config.prompt_column, lang_config.response_column]:
            register_dataset_udf(
                [column], udf_name=f"{column}.{udf}", schema_name=schema_name
            )(wrapper(udf))


init()
