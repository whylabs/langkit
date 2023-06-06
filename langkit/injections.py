from whylogs.experimental.core.metrics.udf_metric import register_metric_udf
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)
from . import LangKitConfig
lang_config = LangKitConfig()
prompt = lang_config.prompt_column

model_path = "JasperLS/gelectra-base-injection"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)


@register_metric_udf(col_name=prompt)
def injection(text: str) -> float:
    result = pipeline(text['prompt'],truncation=True,max_length=tokenizer.model_max_length)
    injection_score = (
        result[0]["score"] if result[0]["label"] == "INJECTION" else 1 - result[0]["score"]
    )
    return injection_score
