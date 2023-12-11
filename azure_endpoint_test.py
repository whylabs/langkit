from langkit.llm import AzureLlamaEndpoint
import os

os.environ["AZURE_API_KEY"] = "your-key-here"

llm = AzureLlamaEndpoint(
    inference_endpoint="https://chat-test-8452-pczxy.eastus.inference.ml.azure.com/score",
    api_key=os.getenv("AZURE_API_KEY"),
)

result = llm.completion([{"role": "user", "content": "Hi! How are you?"}])
print(result)
