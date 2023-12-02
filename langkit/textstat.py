from collections import defaultdict
from logging import getLogger
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
from whylogs.core.stubs import pd
from whylogs.experimental.core.udf_schema import register_dataset_udf
from langkit import LangKitConfig, lang_config, prompt_column, response_column
from langkit.whylogs.unreg import unregister_udfs


diagnostic_logger = getLogger(__name__)


# score metrics

# TODO: should probably s/""/"en"/

# (stat name, language[, udf name])
_udfs_to_register: List[Union[Tuple[str, str], Tuple[str, str, str]]] = [
    ("flesch_kincaid_grade", "text_standard_component"),
    ("flesch_reading_ease", "en"),
    ("smog_index", "text_standard_component"),
    ("coleman_liau_index", "text_standard_component"),
    ("automated_readability_index", "en"),
    ("dale_chall_readability_score", "text_standard_component"),
    ("linsear_write_formula", "text_standard_component"),
    ("gunning_fog", "text_standard_component"),
    ("text_standard", "en", "aggregate_reading_level"),
    ("fernandez_huerta", "es"),
    ("szigriszt_pazos", "es"),
    ("gutierrez_polini", "es"),
    ("crawford", "es"),
    ("gulpease_index", "it"),
    ("osman", "ar"),
    # count metrics
    ("syllable_count", "en"),
    ("lexicon_count", "en"),
    ("sentence_count", "en"),
    ("char_count", "en", "character_count"),
    ("letter_count", "en"),
    ("polysyllabcount", "en", "polysyllable_count"),
    ("monosyllabcount", "en", "monosyllable_count"),
    ("difficult_words", "en"),
]


def wrapper(
    stat_name: str, column: str
) -> Callable[[Union[pd.DataFrame, Dict[str, List]]], Union[pd.Series, List]]:
    import textstat

    stat = textstat.textstat.__getattribute__(stat_name)

    def wrappee(text: Union[pd.DataFrame, Dict[str, List]]) -> Union[pd.Series, List]:
        return [stat(input) for input in text[column]]

    return wrappee


def aggregate_wrapper(
    column: str,
) -> Callable[[Union[pd.DataFrame, Dict[str, List]]], Union[pd.Series, List]]:
    import textstat

    stat = textstat.textstat.text_standard

    def wrappee(text: Union[pd.DataFrame, Dict[str, List]]) -> Union[pd.Series, List]:
        return [stat(input, float_output=True) for input in text[column]]

    return wrappee


def _unpack(t: Union[Tuple[str, str], Tuple[str, str, str]]) -> Tuple[str, str, str]:
    return t if len(t) == 3 else (t[0], t[1], t[0])  # type: ignore


_registered: Dict[str, Set[str]] = defaultdict(
    set
)  # _registered[schema_name] -> set of registered UDF names


def init(
    language: Optional[str] = None,
    config: Optional[LangKitConfig] = None,
    schema_name: str = "",
):
    config = config or lang_config
    prompt_languages = (
        {language} if language is not None else config.prompt_languages
    ) or set()
    response_languages = (
        {language} if language is not None else config.response_languages
    ) or set()
    global _registered
    unregister_udfs(_registered[schema_name], schema_name=schema_name)
    _registered[schema_name] = set()

    for t in _udfs_to_register:
        stat_name, udf_lang, udf = _unpack(t)
        for column in [prompt_column, response_column]:
            if udf_lang in (
                prompt_languages if column == prompt_column else response_languages
            ):
                udf_name = f"{column}.{udf}"
                register_dataset_udf(
                    [column],
                    udf_name=udf_name,
                    schema_name=schema_name,
                )(wrapper(stat_name, column))
                _registered[schema_name].add(udf_name)

    for column in [prompt_column, response_column]:
        if "en" in (
            prompt_languages if column == prompt_column else response_languages
        ):
            udf_name = f"{column}.aggregate_reading_level"
            register_dataset_udf([column], udf_name=udf_name, schema_name=schema_name)(
                aggregate_wrapper(column)
            )
            _registered[schema_name].add(udf_name)

    diagnostic_logger.info("Initialized textstat metrics.")
