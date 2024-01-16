from __future__ import annotations

from typing import Union

from langkit.core.metric import MetricCreator
from langkit.metrics.regexes.regex_loader import CompiledPatternGroups
from langkit.metrics.regexes.regexes import get_custom_substitutions
from langkit.metrics.text_statistics import (
    TextStat,
    prompt_char_count_module,
    prompt_reading_ease_module,
    prompt_textstat_module,
    response_char_count_module,
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

        @staticmethod
        def prompt() -> MetricCreator:
            return prompt_textstat_module

        @staticmethod
        def response() -> MetricCreator:
            return response_textstat_module

        class char_count:
            @staticmethod
            def prompt() -> MetricCreator:
                return prompt_char_count_module

            @staticmethod
            def response() -> MetricCreator:
                return response_char_count_module

        class reading_ease:
            @staticmethod
            # @metric_name("prompt.reading_ease")
            def prompt() -> MetricCreator:
                return prompt_reading_ease_module

            @staticmethod
            # @metric_name("response.reading_ease")
            def response() -> MetricCreator:
                return response_reading_ease_module

            @staticmethod
            def create(text_stat_type: TextStat, input_name: str) -> MetricCreator:
                return lambda: textstat_module(text_stat_type, input_name)

    class substitutions:
        @staticmethod
        def create(input_name: str, file_or_patterns: Union[str, CompiledPatternGroups]) -> MetricCreator:
            return get_custom_substitutions(input_name, file_or_patterns=file_or_patterns)

        @staticmethod
        def prompt(file_or_patterns: Union[str, CompiledPatternGroups]) -> MetricCreator:
            return get_custom_substitutions("prompt", file_or_patterns=file_or_patterns)

        @staticmethod
        def response(file_or_patterns: Union[str, CompiledPatternGroups]) -> MetricCreator:
            return get_custom_substitutions("response", file_or_patterns=file_or_patterns)
