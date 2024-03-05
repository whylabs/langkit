from copy import deepcopy
from typing import Optional

from whylogs.experimental.core.udf_schema import register_dataset_udf
from langkit import LangKitConfig, lang_config, prompt_column, response_column

import os
import torch

_USE_CUDA = torch.cuda.is_available() and not bool(
    os.environ.get("LANGKIT_NO_CUDA", False)
)
_device = 0 if _USE_CUDA else -1

_prompt = prompt_column
_response = response_column
_toxicity_model: Optional["ToxicityModel"] = None


class ToxicityModel:
    def predict(self, text: str) -> float:
        raise NotImplementedError("Subclasses must implement the predict method")


class DetoxifyModel(ToxicityModel):
    def __init__(self, model_name: str):
        from detoxify import Detoxify

        self.detox_model = Detoxify(model_name)

    def predict(self, text: str):
        return self.detox_model.predict(text)["toxicity"]


class ToxicCommentModel(ToxicityModel):
    def __init__(self, model_path: str):
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            TextClassificationPipeline,
        )

        self.toxicity_tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.toxicity_pipeline = TextClassificationPipeline(
            model=model, tokenizer=self.toxicity_tokenizer, device=_device
        )

    def predict(self, text: str) -> float:
        result = self.toxicity_pipeline(
            text, truncation=True, max_length=self.toxicity_tokenizer.model_max_length
        )
        return (
            result[0]["score"]
            if result[0]["label"] == "toxic"
            else 1 - result[0]["score"]
        )


def toxicity(text: str) -> float:
    assert _toxicity_model is not None
    return _toxicity_model.predict(text)


@register_dataset_udf([_prompt], f"{_prompt}.toxicity")
def prompt_toxicity(text):
    return [toxicity(t) for t in text[_prompt]]


@register_dataset_udf([_response], f"{_response}.toxicity")
def response_toxicity(text):
    return [toxicity(t) for t in text[_response]]


def init(model_path: Optional[str] = None, config: Optional[LangKitConfig] = None):
    config = config or deepcopy(lang_config)
    model_path = model_path or config.toxicity_model_path
    global _toxicity_model
    if model_path == "martin-ha/toxic-comment-model":
        _toxicity_model = ToxicCommentModel(model_path)
    elif model_path == "detoxify/unbiased":
        _toxicity_model = DetoxifyModel("unbiased")
    elif model_path == "detoxify/original":
        _toxicity_model = DetoxifyModel("original")
    elif model_path == "detoxify/multilingual":
        _toxicity_model = DetoxifyModel("multilingual")
    else:
        raise ValueError(f"Unknown toxicity model: {model_path}")


init()
