from typing import Optional
import pandas as pd
import json
from logging import getLogger
import pkg_resources


diagnostic_logger = getLogger(__name__)


def show_first_chat(chats: pd.DataFrame):
    print(
        f"prompt: {chats.head(1)['prompt'][0]} response: {chats.head(1)['response'][0]}"
    )
    print()


def load_chats(example_type: Optional[str] = None):
    chats_file_path: str = pkg_resources.resource_filename(
        __name__, "reference_chats.json"
    )
    if example_type is None:
        example_type = "archived_chats"
    results = None
    try:
        with open(chats_file_path, "r") as myfile:
            chats = json.load(myfile)
            results = pd.DataFrame.from_records(chats[example_type])

    except FileNotFoundError:
        diagnostic_logger.warning(f"Could not find {chats_file_path}")
    except json.decoder.JSONDecodeError as json_error:
        diagnostic_logger.warning(f"Could not parse {chats_file_path}: {json_error}")
    return results
