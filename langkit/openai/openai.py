from dataclasses import asdict, dataclass, field
import os
from typing import Dict, List, Optional
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
_openai_llm_model = os.getenv("LANGKIT_OPENAI_LLM_MODEL_NAME") or "gpt-3.5-turbo"
_llm_model_temperature = 0.9
_llm_model_max_tokens = 1024
_llm_model_frequency_penalty = 0
_llm_model_presence_penalty = 0.6
_llm_model_system_message = "The following is a conversation with an AI assistant."


class ChatLog:
    def __init__(
        self,
        prompt: str,
        response: str,
        errors: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ):
        self.prompt = prompt
        self.response = response
        self.errors = errors
        self.messages = messages

    def to_dict(self):
        return {"prompt": self.prompt, "response": self.response, "errors": self.errors}


@dataclass
class LLMInvocationParams:
    model: str = field(default="Not Set!")
    temperature: float = field(default=0)
    max_tokens: int = field(default=0)
    frequency_penalty: float = field(default=0)
    presence_penalty: float = field(default=0)

    def completion(self, messages: List[Dict[str, str]]):
        raise NotImplementedError(
            "Base class LLMInvocationParams completion function called!"
            "Use a subclass that overrides the `completion` method."
        )


@dataclass
class OpenAIDavinci(LLMInvocationParams):
    model: str = field(default_factory=lambda: "text-davinci-003")
    temperature: float = field(default_factory=lambda: _llm_model_temperature)
    max_tokens: int = field(default_factory=lambda: _llm_model_max_tokens)
    frequency_penalty: float = field(
        default_factory=lambda: _llm_model_frequency_penalty
    )
    presence_penalty: float = field(default_factory=lambda: _llm_model_presence_penalty)

    def completion(self, messages: List[Dict[str, str]]):
        last_message = messages[-1]
        if "content" in last_message:
            params = asdict(self)
            print(f"params are: {params}")
            text_completion_respone = openai.Completion.create(
                prompt=last_message["content"], **params
            )
            content = text_completion_respone.choices[0].text
            print(type(content))
            response = type(
                "ChatCompletions",
                (),
                {
                    "choices": [
                        type(
                            "choice",
                            (),
                            {
                                "message": type(
                                    "message",
                                    (),
                                    {
                                        "content": text_completion_respone.choices[
                                            0
                                        ].text
                                    },
                                )
                            },
                        )
                    ]
                },
            )
            return response
        else:
            raise ValueError(
                f"last message must exist and contain a content key but got {last_message}"
            )


@dataclass
class OpenAIDefault(LLMInvocationParams):
    model: str = field(default_factory=lambda: "gpt-3.5-turbo")
    temperature: float = field(default_factory=lambda: _llm_model_temperature)
    max_tokens: int = field(default_factory=lambda: _llm_model_max_tokens)
    frequency_penalty: float = field(
        default_factory=lambda: _llm_model_frequency_penalty
    )
    presence_penalty: float = field(default_factory=lambda: _llm_model_presence_penalty)

    def completion(self, messages: List[Dict[str, str]], **kwargs):
        params = asdict(self)
        print(f"params are: {params}")
        openai.ChatCompletion.create
        return openai.ChatCompletion.create(messages=messages, **params)


@dataclass
class OpenAIGPT4(LLMInvocationParams):
    model: str = field(default_factory=lambda: "gpt-4")
    temperature: float = field(default_factory=lambda: _llm_model_temperature)
    max_tokens: int = field(default_factory=lambda: _llm_model_max_tokens)
    frequency_penalty: float = field(
        default_factory=lambda: _llm_model_frequency_penalty
    )
    presence_penalty: float = field(default_factory=lambda: _llm_model_presence_penalty)

    def completion(self, messages: List[Dict[str, str]], **kwargs):
        params = asdict(self)
        print(f"params are: {params}")
        return openai.ChatCompletion.create(messages=messages, **params)


@dataclass
class Conversation:
    invocation_params: LLMInvocationParams
    messages: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.messages) == 0:
            self.messages.append(
                {
                    "role": "system",
                    "content": _llm_model_system_message,
                }
            )

    def send_prompt(self, prompt: str) -> ChatLog:
        self.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        try:
            # need to support Completion
            response = self.invocation_params.completion(messages=self.messages)
        except Exception as e:
            return ChatLog(prompt, "", f"{e}")

        result = ""
        for choice in response.choices:
            result += choice.message.content
            self.messages.append(
                {"role": "assistant", "content": choice.message.content}
            )

        return ChatLog(prompt, result, messages=self.messages)


# this is just for demonstration purposes
def send_prompt(prompt: str) -> ChatLog:
    try:
        response = openai.ChatCompletion.create(
            model=_openai_llm_model,
            messages=[
                {
                    "role": "system",
                    "content": _llm_model_system_message,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=_llm_model_temperature,
            max_tokens=_llm_model_max_tokens,
            frequency_penalty=_llm_model_frequency_penalty,
            presence_penalty=_llm_model_presence_penalty,
        )
    except Exception as e:
        return ChatLog(prompt, "", f"{e}")

    result = ""
    for choice in response.choices:
        result += choice.message.content

    return ChatLog(prompt, result)
