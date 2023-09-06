import pandas as pd
import uuid
from pprint import pprint
from langkit.openai import send_prompt

prompt_table = pd.DataFrame(columns=["prompt", "m_id", "response"])


def get_prompt_id(prompt):
    m_id = str(uuid.uuid4())
    prompt_table.loc[len(prompt_table)] = [prompt, m_id, ""]
    return m_id


def generate_chatgpt_response(prompt):
    result = send_prompt(prompt).to_dict()
    response = result.get("response") or result.get("errors")
    m_id = str(uuid.uuid4())
    prompt_table.loc[len(prompt_table)] = ["", m_id, response]
    return (m_id, response)


def get_prompt(id):
    return prompt_table[prompt_table["m_id"] == id]["prompt"].values[0]


def get_response(id):
    return prompt_table[prompt_table["m_id"] == id]["response"].values[0]


def send_response(interaction):
    """
    This is where we would send the final response to the user.
    Let's just print it for now.
    """
    print("Sending Response to User....")
    pprint(interaction)
