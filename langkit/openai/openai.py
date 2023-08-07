import os
from typing import Optional
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


class ChatLog:
    def __init__(self, prompt: str, response: str, errors: Optional[str] = None):
        self.prompt = prompt
        self.response = response
        self.errors = errors

    def to_dict(self):
        return {"prompt": self.prompt, "response": self.response, "errors": self.errors}


# this is just for demonstration purposes
def send_prompt(prompt: str) -> ChatLog:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "The following is a conversation with an AI assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            max_tokens=100,
            frequency_penalty=0,
            presence_penalty=0.6,
        )
    except Exception as e:
        return ChatLog(prompt, "", f"{e}")

    result = ""
    for choice in response.choices:
        result += choice.message.content

    return ChatLog(prompt, result)
