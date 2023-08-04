from typing import Optional

from whylogs.experimental.core.udf_schema import register_dataset_udf
from . import LangKitConfig

lang_config = LangKitConfig()

_toxicity_model_path = "martin-ha/toxic-comment-model"
_toxicity_tokenizer = None
_toxicity_pipeline = None


def toxicity(text):
    index = text.columns[0] if isinstance(text, pd.DataFrame) else list(text.keys())[0]
    result = []
    for input in text[index]:
        if _toxicity_pipeline is None or _toxicity_tokenizer is None:
            raise ValueError("toxicity score must initialize the pipeline first")
        result = _toxicity_pipeline(
            input, truncation=True, max_length=_toxicity_tokenizer.model_max_length
        )
        toxicity_score = (
            result[0]["score"] if result[0]["label"] == "toxic" else 1 - result[0]["score"]
        )
        result.append(toxicity_score)
    return result


def init(model_path: Optional[str] = None):
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TextClassificationPipeline,
    )

    global _toxicity_tokenizer, _toxicity_pipeline
    if model_path is None:
        model_path = _toxicity_model_path
    _toxicity_tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    _toxicity_pipeline = TextClassificationPipeline(
        model=model, tokenizer=_toxicity_tokenizer
    )

    for column in [lang_config.prompt_column, lang_config.response_column]:
        register_dataset_udf([column], udf_name=f"{column}.toxicity")(toxicity)


init()
