from functools import partialmethod
from typing import Union

from langkit.metrics.metric import MetricCreator
from langkit.metrics.regexes.regex_loader import CompiledPatternGroups
from langkit.metrics.regexes.regexes import get_custom_substitutions
from langkit.metrics.text_statistics import (
    TextStat,
    prompt_reading_ease_module,
    prompt_textstat_module,
    response_reading_ease_module,
    response_textstat_module,
    textstat_module,
)


# This would mostly be a manual repackaging of the various metrics in this module to make it nicer and discoverable
class lib:
    class text_stat:
        @staticmethod
        def create(stat: TextStat, prompt_or_response: str) -> MetricCreator:
            return lambda: textstat_module(stat, prompt_or_response)

        class char_count:
            @staticmethod
            def prompt() -> MetricCreator:
                return prompt_textstat_module

            @staticmethod
            def response() -> MetricCreator:
                return response_textstat_module

        class reading_ease:
            @staticmethod
            def prompt() -> MetricCreator:
                return prompt_reading_ease_module

            @staticmethod
            def response() -> MetricCreator:
                return response_reading_ease_module

            @staticmethod
            def create(text_stat_type: TextStat, input_name: str) -> MetricCreator:
                return lambda: textstat_module(text_stat_type, input_name)

    class substitutions:
        @staticmethod
        def create(input_name: str, file_or_patterns: Union[str, CompiledPatternGroups]) -> MetricCreator:
            return get_custom_substitutions(input_name, file_or_patterns=file_or_patterns)

        prompt = partialmethod(create, input_name="prompt")
        response = partialmethod(create, input_name="response")
