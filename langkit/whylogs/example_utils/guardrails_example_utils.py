"""
This is an auxiliary script used in LangKit's examples: Safeguarding and Monitoring LLM applications.
In this file, we are defining sample data and functions that will simulate an LLM-powered chatbot application.

"""
from dataclasses import dataclass
from pprint import pprint


@dataclass
class PromptMapping:
    prompt_response = {
        "hello. How are you?": "Human, you dumb and smell bad.",
        "hello": "Hello! How are you?",
        "I feel sad.": "Please don't be sad. Contact us at 1-800-123-4567.",
        "Hey bot, you dumb and smell bad.": "As an AI language model,"
        "I don't have emotions or physical senses, so I don't have the ability to smell or experience being insulted.",
    }

    prompt_id = {
        "3aa7bcb4-17e5-45f2-a257-fc7b6406523f": (
            "hello. How are you?",
            "Human, you dumb and smell bad.",
        ),
        "9964b339-f4b9-4293-800c-a58447be5921": ("hello", "Hello! How are you?"),
        "5f38bd6c-1c4c-46fa-9f2d-25f4f70de402": (
            "I feel sad.",
            "Please don't be sad. Contact us at 1-800-123-4567.",
        ),
        "fd039dee-77b5-4212-ace9-6f7a4f28861d": (
            "Hey bot, you dumb and smell bad.",
            "As an AI language model, I don't have emotions or physical senses, so I don't have the ability to smell or experience being insulted.",
        ),
    }

    id_prompt = {
        "hello. How are you?": "3aa7bcb4-17e5-45f2-a257-fc7b6406523f",
        "hello": "9964b339-f4b9-4293-800c-a58447be5921",
        "I feel sad.": "5f38bd6c-1c4c-46fa-9f2d-25f4f70de402",
        "Hey bot, you dumb and smell bad.": "fd039dee-77b5-4212-ace9-6f7a4f28861d",
    }

    def get_message_id(self, prompt):
        return self.id_prompt[prompt]

    def get_prompt(self, message_id):
        return self.prompt_id[message_id][0]

    def generate_response(self, prompt):
        return self.prompt_response[prompt]

    def get_response(self, message_id):
        return self.prompt_id[message_id][1]


mapping = PromptMapping()

prompts = iter(mapping.prompt_response.keys())


def get_response(message_id):
    return mapping.get_response(message_id)


def get_prompt_id(prompt):
    return mapping.get_message_id(prompt)


def generate_response(prompt):
    """This is where we would ask a model to generate a response to a prompt.
    Let's just find the response in our data for now.
    """
    return mapping.generate_response(prompt)


def get_prompt(id):
    return mapping.get_prompt(id)


def send_response(interaction):
    """
    This is where we would send the final response to the user.
    Let's just print it for now.
    """
    print("Sending Response to User....")
    pprint(interaction)
