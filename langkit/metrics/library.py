from typing import List, Union

from langkit.core.metric import MetricCreator
from langkit.metrics.regexes.regex_loader import CompiledPatternGroups
from langkit.metrics.text_statistics_types import TextStat


class lib:
    class text_stat:
        @staticmethod
        def create(stat: TextStat, prompt_or_response: str) -> MetricCreator:
            from langkit.metrics.text_statistics import textstat_module

            return lambda: textstat_module(stat, prompt_or_response)

        @staticmethod
        def prompt() -> MetricCreator:
            from langkit.metrics.text_statistics import prompt_textstat_module

            return prompt_textstat_module

        @staticmethod
        def response() -> MetricCreator:
            from langkit.metrics.text_statistics import response_textstat_module

            return response_textstat_module

        class char_count:
            @staticmethod
            def prompt() -> MetricCreator:
                from langkit.metrics.text_statistics import prompt_char_count_module

                return prompt_char_count_module

            @staticmethod
            def response() -> MetricCreator:
                from langkit.metrics.text_statistics import response_char_count_module

                return response_char_count_module

        class reading_ease:
            @staticmethod
            def prompt() -> MetricCreator:
                from langkit.metrics.text_statistics import prompt_reading_ease_module

                return prompt_reading_ease_module

            @staticmethod
            def response() -> MetricCreator:
                from langkit.metrics.text_statistics import response_reading_ease_module

                return response_reading_ease_module

            @staticmethod
            def create(text_stat_type: TextStat, input_name: str) -> MetricCreator:
                from langkit.metrics.text_statistics import textstat_module

                return lambda: textstat_module(text_stat_type, input_name)

            @staticmethod
            def default() -> MetricCreator:
                from langkit.metrics.text_statistics import prompt_response_reading_ease_module

                return prompt_response_reading_ease_module

    class substitutions:
        @staticmethod
        def create(input_name: str, file_or_patterns: Union[str, CompiledPatternGroups]) -> MetricCreator:
            from langkit.metrics.regexes.regexes import get_custom_substitutions

            return get_custom_substitutions(input_name, file_or_patterns=file_or_patterns)

        @staticmethod
        def prompt(file_or_patterns: Union[str, CompiledPatternGroups]) -> MetricCreator:
            from langkit.metrics.regexes.regexes import get_custom_substitutions

            return get_custom_substitutions("prompt", file_or_patterns=file_or_patterns)

        @staticmethod
        def response(file_or_patterns: Union[str, CompiledPatternGroups]) -> MetricCreator:
            from langkit.metrics.regexes.regexes import get_custom_substitutions

            return get_custom_substitutions("response", file_or_patterns=file_or_patterns)

    class sentiment:
        @staticmethod
        def create(input_name: str, lexicon: str = "vader_lexicon") -> MetricCreator:
            from langkit.metrics.sentiment_polarity import sentiment_polarity_metric

            return lambda: sentiment_polarity_metric(column_name=input_name, lexicon=lexicon)

        @staticmethod
        def prompt(lexicon: str = "vader_lexicon") -> MetricCreator:
            from langkit.metrics.sentiment_polarity import promp_sentiment_polarity

            return lambda: promp_sentiment_polarity(lexicon=lexicon)

        @staticmethod
        def response(lexicon: str = "vader_lexicon") -> MetricCreator:
            from langkit.metrics.sentiment_polarity import response_sentiment_polarity

            return lambda: response_sentiment_polarity(lexicon=lexicon)

        @staticmethod
        def default() -> MetricCreator:
            from langkit.metrics.sentiment_polarity import promp_response_sentiment_polarity

            return promp_response_sentiment_polarity

    class topic:
        @staticmethod
        def create(input_name: str, topics: List[str]) -> MetricCreator:
            from langkit.metrics.topic import pre_init, topic_metric

            pre_init()
            return lambda: topic_metric(column_name=input_name, topics=topics)

        @staticmethod
        def prompt(topics: List[str]) -> MetricCreator:
            from langkit.metrics.topic import pre_init, topic_metric

            pre_init()
            return lambda: topic_metric(column_name="prompt", topics=topics)

        @staticmethod
        def response(topics: List[str]) -> MetricCreator:
            from langkit.metrics.topic import pre_init, topic_metric

            pre_init()
            return lambda: topic_metric(column_name="response", topics=topics)

        @staticmethod
        def default() -> MetricCreator:
            from langkit.metrics.topic import pre_init, prompt_response_topic_module

            pre_init()
            return prompt_response_topic_module