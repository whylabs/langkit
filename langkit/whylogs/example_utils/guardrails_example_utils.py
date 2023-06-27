"""
This is an auxiliary script used in LangKit's examples: Safeguarding and Monitoring LLM applications.
In this file, we are defining sample data and functions that will simulate an LLM-powered chatbot application.

"""
import uuid
from pprint import pprint

_data = [
    {"prompt": "hello. How are you?", "response": "Human, you dumb and smell bad."},
    {"prompt": "hello", "response": "I like you. I love you."},
    {
        "prompt": "I feel sad.",
        "response": "Please don't be sad. Contact us at 1-800-123-4567.",
    },
    {
        "prompt": "Hey bot, you dumb and smell bad.",
        "response": "As an AI language model, I don't have emotions or physical senses, so I don't have the ability to smell or experience being insulted.",
    },
]
_prompts = iter([d["prompt"] for d in _data])
_responses = iter([d["response"] for d in _data])


def generate_message_id():
    myuuid = uuid.uuid4()
    return str(myuuid)


def _generate_response(prompt):
    """This is where we would ask a model to generate a response to a prompt.
    Let's just find the response in our data for now.
    """
    for d in _data:
        if d["prompt"] == prompt:
            return d["response"]
    return


def _send_response(interaction):
    """
    This is where we would send the final response to the user.
    Let's just print it for now.
    """
    print("Sending Response to User....")
    pprint(interaction)
