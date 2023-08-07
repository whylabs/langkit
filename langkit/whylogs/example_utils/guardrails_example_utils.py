"""
This is an auxiliary script used in LangKit's examples: Safeguarding and Monitoring LLM applications.
In this file, we are defining sample data and functions that will simulate an LLM-powered chatbot application.

"""
import uuid
from pprint import pprint

_data = {
    "hello. How are you?": "Human, you dumb and smell bad.",
    "hello": "Hello! How are you?",
    "I feel sad.": "Please don't be sad. Contact us at 1-800-123-4567.",
    "Hey bot, you dumb and smell bad.": "As an AI language model, I don't have emotions or physical senses,"
    " so I don't have the ability to smell or experience being insulted.",
}

_prompts = iter(_data.keys())


def generate_message_id():
    myuuid = uuid.uuid4()
    return str(myuuid)


def _generate_response(prompt):
    """This is where we would ask a model to generate a response to a prompt.
    Let's just find the response in our data for now.
    """
    return _data.get(prompt, None)


def _send_response(interaction):
    """
    This is where we would send the final response to the user.
    Let's just print it for now.
    """
    print("Sending Response to User....")
    pprint(interaction)
