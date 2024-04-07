# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false
from enum import Enum
from functools import lru_cache
from typing import Any, List, Tuple, cast

import numpy as np
import onnxruntime as ort  # pyright: ignore[reportMissingImports]
import torch
from transformers import BertTokenizerFast

from langkit.asset_downloader import get_asset
from langkit.metrics.embeddings_types import EmbeddingEncoder


@lru_cache
def _get_inference_session(onnx_file_path: str):
    return ort.InferenceSession(onnx_file_path, providers=["CPUExecutionProvider"])  # pyright: ignore[reportUnknownArgumentType]


class TransformerModel(Enum):
    AllMiniLM = ("all-MiniLM-L6-v2", "0")
    ToxicCommentModel = ("toxic-comment-model", "0")

    def cache_model_assets(self):
        """
        Returns the path of the cached model assets, downloading them if necessary.
        """
        name, tag = self.value
        return f"{get_asset(name, tag)}/{name}.onnx"


class OnnxSentenceTransformer(EmbeddingEncoder):
    def __init__(self, model: TransformerModel):
        self._tokenizer: BertTokenizerFast = cast(BertTokenizerFast, BertTokenizerFast.from_pretrained("bert-base-uncased"))
        self._session = _get_inference_session(model.cache_model_assets())

    def encode(self, text: Tuple[str, ...]) -> "torch.Tensor":
        # Pre-truncate the inputs to the model length for better performance
        max_length_in_chars = self._tokenizer.model_max_length * 5  # approx limit
        truncated_text = tuple(content[:max_length_in_chars] for content in text)
        model_inputs = self._tokenizer.batch_encode_plus(list(truncated_text), return_tensors="pt", padding=True, truncation=True)

        input_tensor: torch.Tensor = cast(torch.Tensor, model_inputs["input_ids"])
        inputs_onnx = {"input_ids": input_tensor.cpu().numpy()}
        attention_mask: torch.Tensor = cast(torch.Tensor, model_inputs["attention_mask"])
        inputs_onnx["attention_mask"] = attention_mask.cpu().detach().numpy().astype(np.float32)
        onnx_output: List['np.ndarray["Any", "Any"]'] = cast(List['np.ndarray["Any", "Any"]'], self._session.run(None, inputs_onnx))
        embedding = OnnxSentenceTransformer.mean_pooling(onnx_output=onnx_output, attention_mask=attention_mask)
        return embedding[0]

    @staticmethod
    def mean_pooling(
        onnx_output: List['np.ndarray["Any", "Any"]'], attention_mask: torch.Tensor
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        token_embeddings = torch.from_numpy(onnx_output[0])
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask, input_mask_expanded, sum_mask
