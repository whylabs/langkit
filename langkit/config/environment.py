import openai
import getpass
import os


def check_openai_api_key():
    open_api_key = os.getenv("OPENAI_API_KEY")
    if open_api_key is None:
        print("Enter your OPENAI_APIKEY", flush=True)
        os.environ["OPENAI_API_KEY"] = getpass.getpass()
        print("OPENAI_API_KEY set!")
    else:
        print("OPENAI_API_KEY already set in env var, good job!")
    openai.api_key = os.getenv("OPENAI_API_KEY")


def check_or_prompt_for_api_keys():
    # set your org-id here - should be something like "org-xxxx"
    org_id = os.environ.get("WHYLABS_DEFAULT_ORG_ID")
    if org_id is None:
        print("Enter your WhyLabs Org ID", flush=True)
        os.environ["WHYLABS_DEFAULT_ORG_ID"] = input()
        org_id = os.environ.get("WHYLABS_DEFAULT_ORG_ID")
    else:
        print(f"WhyLabs Org ID is already set in env var to: {org_id}")

    dataset_id = os.environ.get("WHYLABS_DEFAULT_DATASET_ID")
    if dataset_id is None:
        # set your datased_id (or model_id) here - should be something like "model-xxxx"
        print("Enter your WhyLabs Dataset ID", flush=True)
        os.environ["WHYLABS_DEFAULT_DATASET_ID"] = input()
        dataset_id = os.environ.get("WHYLABS_DEFAULT_DATASET_ID")
    else:
        print(f"WhyLabs Dataset ID is already set in env var to: {dataset_id}")

    # set your API key here
    whylabs_api_key = os.environ.get("WHYLABS_API_KEY")
    if whylabs_api_key is None:
        print("Enter your WhyLabs API key", flush=True)
        os.environ["WHYLABS_API_KEY"] = getpass.getpass()
        print("Using API Key ID: ", os.environ["WHYLABS_API_KEY"][0:10])
    else:
        print(
            "Whylabs API Key already set with ID: ", os.environ["WHYLABS_API_KEY"][0:10]
        )

    check_openai_api_key()
