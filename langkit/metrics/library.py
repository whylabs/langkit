from typing import List, Optional, Union

from langkit.core.metric import MetricCreator
from langkit.metrics.regexes.regex_loader import CompiledPatternGroups
from langkit.metrics.text_statistics_types import TextStat


class lib:
    @staticmethod
    def all_metrics() -> MetricCreator:
        from langkit.metrics.injections import prompt_injections_module
        from langkit.metrics.input_output_similarity import prompt_response_input_output_similarity_module
        from langkit.metrics.pii import prompt_presidio_pii_metric, response_presidio_pii_metric
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
        from langkit.metrics.themes.themes import prompt_jailbreak_similarity_metric, response_refusal_similarity_metric
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
            prompt_injections_module,
            prompt_jailbreak_similarity_metric,
            response_refusal_similarity_metric,
            prompt_presidio_pii_metric,
            response_presidio_pii_metric,
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

    class injections:
        @staticmethod
        def prompt() -> MetricCreator:
            from langkit.metrics.injections import prompt_injections_module

            return prompt_injections_module

        @staticmethod
        def create(input_name: str) -> MetricCreator:
            from langkit.metrics.injections import injections_metric

            return lambda: injections_metric(column_name=input_name)

    class input_output:
        @staticmethod
        def similarity(input_column_name: str = "prompt", output_column_name: str = "response") -> MetricCreator:
            from langkit.metrics.input_output_similarity import input_output_similarity_metric

            return lambda: input_output_similarity_metric(input_column_name=input_column_name, output_column_name=output_column_name)

        @staticmethod
        def default() -> MetricCreator:
            from langkit.metrics.input_output_similarity import prompt_response_input_output_similarity_module

            return prompt_response_input_output_similarity_module

    class jailbreak:
        @staticmethod
        def prompt() -> MetricCreator:
            from langkit.metrics.themes.themes import prompt_jailbreak_similarity_metric

            return prompt_jailbreak_similarity_metric

    class refusal:
        @staticmethod
        def response() -> MetricCreator:
            from langkit.metrics.themes.themes import response_refusal_similarity_metric

            return response_refusal_similarity_metric

    class pii_presidio:
        @staticmethod
        def create(
            input_name: str,
            language: str = "en",
            spacy_model: str = "en_core_web_sm",
            transformers_model: str = "dslim/bert-base-NER",
            entities: Optional[List[str]] = None,
        ) -> MetricCreator:
            """
            Create a PII metric using the Presidio analyzer.

            :param input_name: The name of the input column.
            :param language: The language to use for the analyzer.
            :param spacy_model: The spaCy model to use for the analyzer.
            :param transformers_model: The transformers model to use for the analyzer.
            :param entities: The list of entities to analyze for. See https://microsoft.github.io/presidio/supported_entities/.
            :return: A metric creator.
            """
            from langkit.metrics.pii import pii_presidio_metric

            return lambda: pii_presidio_metric(input_name, language, spacy_model, transformers_model, entities)

        @staticmethod
        def prompt() -> MetricCreator:
            from langkit.metrics.pii import prompt_presidio_pii_metric

            return prompt_presidio_pii_metric

        @staticmethod
        def response() -> MetricCreator:
            from langkit.metrics.pii import response_presidio_pii_metric

            return response_presidio_pii_metric

        @staticmethod
        def default() -> MetricCreator:
            from langkit.metrics.pii import prompt_response_presidio_pii_metric

            return prompt_response_presidio_pii_metric
