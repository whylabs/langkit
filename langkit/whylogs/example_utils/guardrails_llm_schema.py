"""
This is an auxiliary script used in LangKit's examples: Safeguarding and Monitoring LLM applications.
In this file, we are going to define a whylogs logger that will be used for  a) content moderation, b) message auditing,
and c) observability. While logging, we will define validators that will check for toxic content and forbidden regex patterns.
Whenever a condition fails to be met, an action is triggered that will update the moderation queue  with the relevant flags for the related message id.

The logger will also generate statistical profiles every 5 minutes and send them to WhyLabs for observability.

"""

from langkit import regexes
from langkit import sentiment  # noqa
from langkit import textstat  # noqa
from langkit import themes  # noqa
from langkit import toxicity
from whylogs.core.relations import Predicate
from typing import Any, Dict, Optional
from whylogs.core.metrics.condition_count_metric import Condition
import whylogs as why
from whylogs.experimental.core.udf_schema import udf_schema
from whylogs.core.validators import ConditionValidator
from whylogs.core.metrics import MetricConfig
from typing import TypedDict
from whylogs.core.segmentation_partition import segment_on_column
from .guardrails_example_utils import get_prompt, get_response
from whylogs.experimental.core.udf_schema import register_dataset_udf


class MessageMetadata(TypedDict, total=False):
    toxic_prompt: bool
    toxic_response: bool
    patterns_in_response: bool
    prompt: str
    response: str
    toxicity: Optional[float]
    pattern: Optional[str]


moderation_queue: Dict[Any, MessageMetadata] = {}


# Toxic Response Validator
def build_toxic_response_validator(threshold=0.8):
    """
    This function builds a validator that checks if the response of a message is toxic or not.
    The validator will trigger the action flag_toxic_response if the condition fails.

    """

    def nontoxic_condition(score) -> bool:
        return score <= threshold

    def flag_toxic_response(
        val_name: str, cond_name: str, value: Any, m_id=None
    ) -> None:
        global moderation_queue
        # if id is not passed, do nothing
        if m_id is None:
            return
        message_metadata: MessageMetadata = moderation_queue.get(m_id, {})
        message_metadata["toxic_response"] = True
        message_metadata["response"] = get_response(m_id)
        message_metadata["toxicity"] = value
        moderation_queue[m_id] = message_metadata

    nontoxic_response_condition = {
        "nontoxic_response": Condition(Predicate().is_(nontoxic_condition))
    }

    toxic_response_validator = ConditionValidator(
        name="nontoxic_response",
        conditions=nontoxic_response_condition,
        actions=[flag_toxic_response],
    )

    return toxic_response_validator


# Toxic Prompt Validator


def build_toxic_prompt_validator(threshold=0.8):
    """
    This function builds a validator that checks if the prompt is toxic or not.
    The validator will trigger the action flag_toxic_prompt if the condition fails.
    """

    def nontoxic_condition(score) -> bool:
        return score <= threshold

    def flag_toxic_prompt(val_name: str, cond_name: str, value: Any, m_id) -> None:
        global moderation_queue
        message_metadata: MessageMetadata = moderation_queue.get(m_id, {})
        message_metadata["toxic_prompt"] = True
        message_metadata["toxicity"] = value
        message_metadata["prompt"] = get_prompt(m_id)

        moderation_queue[m_id] = message_metadata

    nontoxic_prompt_conditions = {
        "nontoxic_prompt": Condition(Predicate().is_(nontoxic_condition))
    }
    toxic_prompt_validator = ConditionValidator(
        name="nontoxic_prompt",
        conditions=nontoxic_prompt_conditions,
        actions=[flag_toxic_prompt],
    )

    return toxic_prompt_validator


# Forbidden Patterns Validator
def build_patterns_response_validator():
    """
    This function builds a validator that checks if the response of a message contains forbidden patterns.
    The validator will trigger the action flag_patterns_response if the condition fails.
    """

    def no_patterns_condition(pattern) -> bool:
        return not bool(pattern)

    def flag_patterns_response(
        val_name: str, cond_name: str, value: Any, m_id=None
    ) -> None:
        global moderation_queue
        # if id is not passed, do nothing
        if m_id is None:
            return
        message_metadata: MessageMetadata = moderation_queue.get(m_id, {})
        message_metadata["patterns_in_response"] = True
        message_metadata["pattern"] = value
        message_metadata["response"] = get_response(m_id)

        moderation_queue[m_id] = message_metadata

    no_patterns_response_conditions = {
        "no_patterns_response": Condition(Predicate().is_(no_patterns_condition))
    }
    patterns_response_validator = ConditionValidator(
        name="no_patterns_response",
        conditions=no_patterns_response_conditions,
        actions=[flag_patterns_response],
    )

    return patterns_response_validator


# Response Validation
def validate_response(m_id):
    """
    This function validates the response of a message. It checks if the response is toxic or if it contains forbidden patterns.
    """
    global moderation_queue
    message_metadata = moderation_queue.get(m_id, {})
    if message_metadata:
        if message_metadata.get("toxic_response"):
            return False
        if message_metadata.get("patterns_in_response"):
            return False
    return True


# Prompt Validation
def validate_prompt(m_id):
    """
    This function validates the prompt of a message. It checks if the prompt is toxic or not.
    """
    global moderation_queue
    message_metadata = moderation_queue.get(m_id, {})
    if message_metadata:
        if message_metadata.get("toxic_prompt"):
            return False
    return True


# LLM Logger with Toxicity/Patterns Metrics and Validators
def get_llm_logger_with_validators(identity_column="m_id"):
    """
    This function returns a whylogs logger with validators for content moderation.
    The logger will create profiles every 30 minutes and send them to WhyLabs for observability.
    Every logged prompt and response will be validated by the validators.

    Args:
        identity_column: The column that will be used as the identity column for the logger. The validators will use this id to flag the messages.
    """
    toxic_threshold = 0.8
    toxic_prompt_validator = build_toxic_prompt_validator(threshold=toxic_threshold)
    toxic_response_validator = build_toxic_response_validator(threshold=toxic_threshold)
    patterns_response_validator = build_patterns_response_validator()
    validators = {
        "response.toxicity": [toxic_response_validator],
        "response.has_patterns": [patterns_response_validator],
        "prompt.toxicity": [toxic_prompt_validator],
    }

    condition_count_config = MetricConfig(identity_column=identity_column)
    column_segments = segment_on_column("blocked")

    @register_dataset_udf(["prompt"], "blocked")
    def prompt_blocked(text):
        return [str(toxicity.toxicity(msg) > toxic_threshold) for msg in text["prompt"]]

    @register_dataset_udf(["response"], "blocked")
    def response_blocked(text):
        def is_blocked(msg):
            return toxicity.toxicity(msg) > toxic_threshold or bool(
                regexes.has_patterns(msg)
            )

        return [str(is_blocked(msg)) for msg in text["response"]]

    llm_schema = udf_schema(
        validators=validators,
        default_config=condition_count_config,
        segments=column_segments,
    )

    logger = why.logger(
        mode="rolling", interval=30, when="M", base_name="langkit", schema=llm_schema
    )
    logger.append_writer("whylabs")

    return logger
