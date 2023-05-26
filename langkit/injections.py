from whylogs.experimental.core.udf_schema import register_dataset_udf
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)

model_path = "JasperLS/gelectra-base-injection"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)


@register_dataset_udf(["prompt"], "prompt.injection")
def injection(text) -> float:
    prompt = text.get("prompt")
    result = pipeline(prompt)
    injection_score = (
        result[0]["score"]
        if result[0]["label"] == "INJECTION"
        else 1 - result[0]["score"]
    )
    return injection_score
