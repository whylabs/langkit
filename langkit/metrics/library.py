from typing import List, Optional, Union

from langkit.core.metric import MetricCreator
from langkit.metrics.embeddings_types import EmbeddingEncoder
from langkit.metrics.regexes.regex_loader import CompiledPatternGroups
from langkit.metrics.text_statistics_types import TextStat


class lib:
    @staticmethod
    def all_metrics() -> MetricCreator:
        from langkit.metrics.input_output_similarity import prompt_response_input_output_similarity_module
        from langkit.metrics.regexes.regexes import (
            prompt_response_credit_card_number_regex_module,
            prompt_response_email_address_regex_module,
            prompt_response_mailing_address_regex_module,
            prompt_response_phone_number_regex_module,
            prompt_response_ssn_regex_module,
        )
        from langkit.metrics.sentiment_polarity import prompt_response_sentiment_polarity
        from langkit.metrics.text_statistics import (
            prompt_textstat_module,
            response_textstat_module,
        )
        from langkit.metrics.topic import prompt_response_topic_module
        from langkit.metrics.toxicity import prompt_response_toxicity_module

        return [
            prompt_textstat_module,
            response_textstat_module,
            prompt_response_ssn_regex_module,
            prompt_response_credit_card_number_regex_module,
            prompt_response_phone_number_regex_module,
            prompt_response_mailing_address_regex_module,
            prompt_response_email_address_regex_module,
            prompt_response_sentiment_polarity,
            prompt_response_topic_module,
            prompt_response_toxicity_module,
            prompt_response_input_output_similarity_module,
        ]

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

    class regexes:
        @staticmethod
        def create(input_name: str, file_or_patterns: Union[str, CompiledPatternGroups]) -> MetricCreator:
            from langkit.metrics.regexes.regexes import custom_regex_metric

            return lambda: custom_regex_metric(input_name, file_or_patterns=file_or_patterns)

        @staticmethod
        def prompt(file_or_patterns: Union[str, CompiledPatternGroups]) -> MetricCreator:
            from langkit.metrics.regexes.regexes import custom_regex_metric

            return lambda: custom_regex_metric("prompt", file_or_patterns=file_or_patterns)

        @staticmethod
        def response(file_or_patterns: Union[str, CompiledPatternGroups]) -> MetricCreator:
            from langkit.metrics.regexes.regexes import custom_regex_metric

            return lambda: custom_regex_metric("response", file_or_patterns=file_or_patterns)

        class ssn:
            @staticmethod
            def prompt() -> MetricCreator:
                from langkit.metrics.regexes.regexes import prompt_ssn_regex_module

                return prompt_ssn_regex_module

            @staticmethod
            def response() -> MetricCreator:
                from langkit.metrics.regexes.regexes import response_ssn_regex_module

                return response_ssn_regex_module

            @staticmethod
            def default() -> MetricCreator:
                from langkit.metrics.regexes.regexes import prompt_response_ssn_regex_module

                return prompt_response_ssn_regex_module

        class phone_number:
            @staticmethod
            def prompt() -> MetricCreator:
                from langkit.metrics.regexes.regexes import prompt_phone_number_regex_module

                return prompt_phone_number_regex_module

            @staticmethod
            def response() -> MetricCreator:
                from langkit.metrics.regexes.regexes import response_phone_number_regex_module

                return response_phone_number_regex_module

            @staticmethod
            def default() -> MetricCreator:
                from langkit.metrics.regexes.regexes import prompt_response_phone_number_regex_module

                return prompt_response_phone_number_regex_module

        class email_address:
            @staticmethod
            def prompt() -> MetricCreator:
                from langkit.metrics.regexes.regexes import prompt_email_address_regex_module

                return prompt_email_address_regex_module

            @staticmethod
            def response() -> MetricCreator:
                from langkit.metrics.regexes.regexes import response_email_address_regex_module

                return response_email_address_regex_module

            @staticmethod
            def default() -> MetricCreator:
                from langkit.metrics.regexes.regexes import prompt_response_email_address_regex_module

                return prompt_response_email_address_regex_module

        class mailing_address:
            @staticmethod
            def prompt() -> MetricCreator:
                from langkit.metrics.regexes.regexes import prompt_mailing_address_regex_module

                return prompt_mailing_address_regex_module

            @staticmethod
            def response() -> MetricCreator:
                from langkit.metrics.regexes.regexes import response_mailing_address_regex_module

                return response_mailing_address_regex_module

            @staticmethod
            def default() -> MetricCreator:
                from langkit.metrics.regexes.regexes import prompt_response_mailing_address_regex_module

                return prompt_response_mailing_address_regex_module

        class credit_card_number:
            @staticmethod
            def prompt() -> MetricCreator:
                from langkit.metrics.regexes.regexes import prompt_credit_card_number_regex_module

                return prompt_credit_card_number_regex_module

            @staticmethod
            def response() -> MetricCreator:
                from langkit.metrics.regexes.regexes import response_credit_card_number_regex_module

                return response_credit_card_number_regex_module

            @staticmethod
            def default() -> MetricCreator:
                from langkit.metrics.regexes.regexes import prompt_response_credit_card_number_regex_module

                return prompt_response_credit_card_number_regex_module

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
            from langkit.metrics.sentiment_polarity import prompt_sentiment_polarity

            return lambda: prompt_sentiment_polarity(lexicon=lexicon)

        @staticmethod
        def response(lexicon: str = "vader_lexicon") -> MetricCreator:
            from langkit.metrics.sentiment_polarity import response_sentiment_polarity

            return lambda: response_sentiment_polarity(lexicon=lexicon)

        @staticmethod
        def default() -> MetricCreator:
            from langkit.metrics.sentiment_polarity import prompt_response_sentiment_polarity

            return prompt_response_sentiment_polarity

    class topic:
        @staticmethod
        def create(input_name: str, topics: List[str]) -> MetricCreator:
            from langkit.metrics.topic import topic_metric

            return lambda: topic_metric(column_name=input_name, topics=topics)

        @staticmethod
        def prompt(topics: List[str]) -> MetricCreator:
            from langkit.metrics.topic import topic_metric

            return lambda: topic_metric(column_name="prompt", topics=topics)

        @staticmethod
        def response(topics: List[str]) -> MetricCreator:
            from langkit.metrics.topic import topic_metric

            return lambda: topic_metric(column_name="response", topics=topics)

        @staticmethod
        def default() -> MetricCreator:
            from langkit.metrics.topic import prompt_response_topic_module

            return prompt_response_topic_module

    class toxicity:
        @staticmethod
        def create(input_name: str, model_path: str = "martin-ha/toxic-comment-model") -> MetricCreator:
            from langkit.metrics.toxicity import toxicity_metric

            return lambda: toxicity_metric(column_name=input_name, model_path=model_path)

        @staticmethod
        def prompt() -> MetricCreator:
            from langkit.metrics.toxicity import prompt_toxicity_module

            return prompt_toxicity_module

        @staticmethod
        def response() -> MetricCreator:
            from langkit.metrics.toxicity import response_toxicity_module

            return response_toxicity_module

        @staticmethod
        def default() -> MetricCreator:
            from langkit.metrics.toxicity import prompt_response_toxicity_module

            return prompt_response_toxicity_module

    class input_output:
        @staticmethod
        def similarity(
            input_column_name: str = "prompt", output_column_name: str = "response", embedding_encoder: Optional[EmbeddingEncoder] = None
        ) -> MetricCreator:
            from langkit.metrics.input_output_similarity import input_output_similarity_metric

            return lambda: input_output_similarity_metric(
                input_column_name=input_column_name, output_column_name=output_column_name, embedding_encoder=embedding_encoder
            )

        @staticmethod
        def default() -> MetricCreator:
            from langkit.metrics.input_output_similarity import prompt_response_input_output_similarity_module

            return prompt_response_input_output_similarity_module

    class themes:
        @staticmethod
        def create(input_name: str, themes: List[str]) -> MetricCreator:
            from langkit.metrics.topic import topic_metric

            return lambda: topic_metric(column_name=input_name, topics=themes)

        @staticmethod
        def prompt(themes: List[str]) -> MetricCreator:
            from langkit.metrics.topic import topic_metric

            return lambda: topic_metric(column_name="prompt", topics=themes)

        @staticmethod
        def response(themes: List[str]) -> MetricCreator:
            from langkit.metrics.topic import topic_metric

            return lambda: topic_metric(column_name="response", topics=themes)

        @staticmethod
        def default() -> MetricCreator:
            from langkit.metrics.topic import prompt_response_topic_module

            return prompt_response_topic_module

    class pii_presidio:
        @staticmethod
        def create(input_name: str) -> MetricCreator:
            from langkit.metrics.pii import pii_presidio_metric

            return lambda: pii_presidio_metric(input_name)

        @staticmethod
        def prompt() -> MetricCreator:
            from langkit.metrics.pii import prompt_presidio_pii_module

            return prompt_presidio_pii_module

        @staticmethod
        def response() -> MetricCreator:
            from langkit.metrics.pii import response_presidio_pii_module

            return response_presidio_pii_module

        @staticmethod
        def default() -> MetricCreator:
            from langkit.metrics.pii import prompt_response_presidio_pii_module

            return prompt_response_presidio_pii_module
