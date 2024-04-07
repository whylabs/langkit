# pyright: reportUnknownMemberType=none
# pyright: reportUnknownVariableType=none
# pyright: reportUnknownLambdaType=none
import os
from functools import lru_cache, partial
from typing import List, cast

import numpy as np
import onnxruntime
import pandas as pd
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from langkit.asset_downloader import get_asset
from langkit.core.metric import Metric, SingleMetric, SingleMetricResult, UdfInput
from langkit.onnx_encoder import TransformerModel


def __toxicity(tokenizer: PreTrainedTokenizerBase, session: onnxruntime.InferenceSession, max_length: int, text: List[str]) -> List[float]:
    max_length_in_chars = tokenizer.model_max_length * 5
    truncated_text = [content[:max_length_in_chars] for content in text]
    inputs = tokenizer(truncated_text, return_tensors="pt", padding=True, truncation=True)
    onnx_inputs = {k: v.numpy() for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
    onnx_output_logits = session.run(None, onnx_inputs)[0]

    # Apply softmax to convert logits into probabilities
    probabilities = np.exp(onnx_output_logits) / np.sum(np.exp(onnx_output_logits), axis=1, keepdims=True)  # pyright: ignore[reportUnknownArgumentType]
    labels = ["non-toxic", "toxic"]
    # Find the index of the highest probability to determine the predicted label
    predicted_label_idx = np.argmax(probabilities, axis=1)
    predicted_labels: List[str] = [labels[idx] for idx in predicted_label_idx]
    predicted_scores: List[float] = [prob[idx] for prob, idx in zip(probabilities, predicted_label_idx)]
    results = [{"label": label, "score": score} for label, score in zip(predicted_labels, predicted_scores)]
    return [result["score"] if result["label"] == "toxic" else 1.0 - result["score"] for result in results]  # type: ignore


def _download_assets():
    name, tag = TransformerModel.ToxicCommentModel.value
    return get_asset(name, tag)


@lru_cache
def _get_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(_download_assets())


@lru_cache
def _get_session() -> onnxruntime.InferenceSession:
    downloaded_path = _download_assets()
    onnx_model_path = os.path.join(downloaded_path, "model.onnx")
    print(f"Loading ONNX model from {onnx_model_path}")
    return onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])


def toxicity_metric(column_name: str) -> Metric:
    def cache_assets():
        _download_assets()

    def init():
        _get_session()
        _get_tokenizer()

    def udf(text: pd.DataFrame) -> SingleMetricResult:
        _tokenizer = _get_tokenizer()
        _session = _get_session()

        col = list(UdfInput(text).iter_column_rows(column_name))
        max_length = cast(int, _tokenizer.model_max_length)
        metrics = __toxicity(_tokenizer, _session, max_length, col)
        return SingleMetricResult(metrics=metrics)

    return SingleMetric(
        name=f"{column_name}.toxicity.toxicity_score", input_names=[column_name], evaluate=udf, init=init, cache_assets=cache_assets
    )


prompt_toxicity_metric = partial(toxicity_metric, "prompt")
response_toxicity_metric = partial(toxicity_metric, "response")
prompt_response_toxicity_module = [prompt_toxicity_metric, response_toxicity_metric]
