# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false
import time
from enum import Enum
from functools import lru_cache
from os import environ
from typing import Any, List, Tuple, cast

import numpy as np
import onnxruntime as ort  # pyright: ignore[reportMissingImports]
import torch
from psutil import cpu_count
from transformers import BertTokenizerFast

from langkit.asset_downloader import get_asset
from langkit.metrics.embeddings_types import EmbeddingEncoder


@lru_cache
def _get_inference_session(onnx_file_path: str):
    cpus = cpu_count(logical=True)
    # environ["OMP_NUM_THREADS"] = str(cpus)
    # environ["OMP_WAIT_POLICY"] = "ACTIVE"
    sess_opts: ort.SessionOptions = ort.SessionOptions()
    # sess_opts.enable_cpu_mem_arena = True
    # sess_opts.inter_op_num_threads = cpus
    # sess_opts.intra_op_num_threads = 1
    # sess_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    # sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # sess_opts.enable_mem_pattern = True

    return ort.InferenceSession(onnx_file_path, providers=["CPUExecutionProvider"], sess_options=sess_opts)  # pyright: ignore[reportUnknownArgumentType]


class TransformerModel(Enum):
    AllMiniLM = ("all-MiniLM-L6-v2", "0")

    def get_model_path(self):
        name, tag = self.value
        return f"{get_asset(name, tag)}/{name}.onnx"


# _times: List[float] = []


class OnnxSentenceTransformer(EmbeddingEncoder):
    def __init__(self, model: TransformerModel):
        self._tokenizer: BertTokenizerFast = cast(BertTokenizerFast, BertTokenizerFast.from_pretrained("bert-base-uncased"))
        self._model = model
        self._session = _get_inference_session(model.get_model_path())

    def encode(self, text: Tuple[str, ...]) -> "torch.Tensor":
        model_inputs = self._tokenizer.batch_encode_plus(list(text), return_tensors="pt", padding=True, truncation=True)
        input_tensor: torch.Tensor = cast(torch.Tensor, model_inputs["input_ids"])
        inputs_onnx = {"input_ids": input_tensor.cpu().numpy()}
        attention_mask: torch.Tensor = cast(torch.Tensor, model_inputs["attention_mask"])
        inputs_onnx["attention_mask"] = attention_mask.cpu().detach().numpy().astype(np.float32)
        start_time = time.perf_counter()
        onnx_output: List['np.ndarray["Any", "Any"]'] = cast(List['np.ndarray["Any", "Any"]'], self._session.run(None, inputs_onnx))
        # _times.append(time.perf_counter() - start_time)
        # print(f"Average time: {sum(_times) / len(_times)}")
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
