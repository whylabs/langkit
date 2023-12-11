from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import urllib.request
import json
import os
import ssl

_llm_model_temperature = 0.4
_llm_max_length = 200
_llm_max_new_tokens = 200


def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if (
        allowed
        and not os.environ.get("PYTHONHTTPSVERIFY", "")
        and getattr(ssl, "_create_unverified_context", None)
    ):
        ssl._create_default_https_context = ssl._create_unverified_context


allowSelfSignedHttps(
    True
)  # this line is needed if you use self-signed certificate in your scoring service.


@dataclass
class AzureLlamaEndpoint:
    inference_endpoint: str
    api_key: str
    temperature: float = field(default_factory=lambda: _llm_model_temperature)
    max_length: int = field(default_factory=lambda: _llm_max_length)
    max_new_tokens: int = field(default_factory=lambda: _llm_max_new_tokens)
    model_kwargs: Optional[dict] = None

    def completion(self, messages: List[Dict[str, str]], **kwargs):
        data = {
            "input_data": {
                "input_string": messages,
                "parameters": self.model_kwargs
                or {
                    "temperatue": self.temperature,
                    "max_length": self.max_length,
                    "max_new_tokens": self.max_new_tokens,
                },
            },
        }
        body = str.encode(json.dumps(data))
        headers = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self.api_key),
            "azureml-model-deployment": "llama-2-13b-chat-13",
        }

        req = urllib.request.Request(self.inference_endpoint, body, headers)

        try:
            response = urllib.request.urlopen(req)

            result = response.read()
            return result
        except urllib.error.HTTPError as error:
            print("The request failed with status code: " + str(error.code))

            # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
            print(error.info())
            print(error.read().decode("utf8", "ignore"))
