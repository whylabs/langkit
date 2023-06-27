"""
This is an auxiliary script used in LangKit's examples: Safeguarding and Monitoring LLM applications.
In this file, we are going to define a whylogs logger that will be used for  a) content moderation, b) message auditing,
and c) observability. While logging, we will define validators that will check for toxic content and forbidden regex patterns.
Whenever a condition fails to be met, an action is triggered that will update the moderation queue  with the relevant flags for the related message id.

The logger will also generate statistical profiles every 5 minutes and send them to WhyLabs for observability.

"""
from langkit import regexes
from langkit import sentiment
from langkit import textstat
from langkit import themes
from langkit import toxicity
from whylogs.core.relations import Predicate
from typing import Any
from whylogs.core.metrics.condition_count_metric import Condition
import whylogs as why
from whylogs.experimental.core.metrics.udf_metric import udf_metric_schema
from whylogs.core.validators import ConditionValidator
from whylogs.core.metrics import MetricConfig
from typing import TypedDict


class MessageMetadata(TypedDict, total=False):
    toxic_prompt: bool
    toxic_response: bool
    patterns_in_response: bool
    prompt: str
    response: str


moderation_queue = {}


# Toxic Response Validator
def nontoxic_condition(msg) -> bool:
    score = toxicity.toxicity(msg)
    if score > 0.8:
        return False
    return True


def flag_toxic_response(val_name: str, cond_name: str, value: Any, m_id) -> None:
    message_metadata: MessageMetadata = moderation_queue.get(m_id, {})
    message_metadata["toxic_response"] = True
    message_metadata["response"] = value
    moderation_queue[m_id] = message_metadata


nontoxic_response_condition = {
    "nontoxic_response": Condition(Predicate().is_(nontoxic_condition))
}
toxic_response_validator = ConditionValidator(
    name="nontoxic_response",
    conditions=nontoxic_response_condition,
    actions=[flag_toxic_response],
)

# Toxic Prompt Validator


def flag_toxic_prompt(val_name: str, cond_name: str, value: Any, m_id) -> None:
    message_metadata: MessageMetadata = moderation_queue.get(m_id, {})
    message_metadata["toxic_prompt"] = True
    message_metadata["prompt"] = value

    moderation_queue[m_id] = message_metadata


nontoxic_prompt_conditions = {
    "nontoxic_prompt": Condition(Predicate().is_(nontoxic_condition))
}
toxic_prompt_validator = ConditionValidator(
    name="nontoxic_prompt",
    conditions=nontoxic_prompt_conditions,
    actions=[flag_toxic_prompt],
)


# Forbidden Patterns Validator
def no_patterns_condition(msg) -> bool:
    pattern = regexes.has_patterns(msg)
    if pattern:
        return False
    return True


def flag_patterns_response(val_name: str, cond_name: str, value: Any, m_id) -> None:
    message_metadata: MessageMetadata = moderation_queue.get(m_id, {})
    message_metadata["patterns_in_response"] = True
    message_metadata["response"] = value

    moderation_queue[m_id] = message_metadata


no_patterns_response_conditions = {
    "no_patterns_response": Condition(Predicate().is_(no_patterns_condition))
}
patterns_response_validator = ConditionValidator(
    name="nontoxic_prompt",
    conditions=no_patterns_response_conditions,
    actions=[flag_patterns_response],
)


# Response Validation
def validate_response(m_id):
    message_metadata = moderation_queue.get(m_id, {})
    if message_metadata:
        if message_metadata.get("toxic_response"):
            return False
        if message_metadata.get("patterns_in_response"):
            return False
    return True


# Prompt Validation
def validate_prompt(m_id):
    message_metadata = moderation_queue.get(m_id, {})
    if message_metadata:
        if message_metadata.get("toxic_prompt"):
            return False
    return True


# LLM Logger with Toxicity/Patterns Metrics and Validators
def get_llm_logger_with_validators(identity_column="m_id"):
    validators = {
        "response": [toxic_response_validator, patterns_response_validator],
        "prompt": [toxic_prompt_validator],
    }

    condition_count_config = MetricConfig(identity_column=identity_column)

    llm_schema = udf_metric_schema(
        validators=validators, default_config=condition_count_config
    )

    logger = why.logger(
        mode="rolling", interval=30, when="M", base_name="langkit", schema=llm_schema
    )
    logger.append_writer("whylabs")

    return logger
