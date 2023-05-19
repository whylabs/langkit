from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)
from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import register_metric_udf

model_path = "martin-ha/toxic-comment-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)


@register_metric_udf(col_type=String)
def toxicity(text: str) -> float:
    result = pipeline(text)
    toxicity_score = (
        result[0]["score"] if result[0]["label"] == "toxic" else 1 - result[0]["score"]
    )
    return toxicity_score
