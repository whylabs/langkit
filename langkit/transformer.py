from importlib import resources
from logging import getLogger
from sentence_transformers import SentenceTransformer
from typing import Optional, Callable, Union, List, Any
from torch import Tensor
import numpy as np

import os
import torch

_USE_CUDA = torch.cuda.is_available() and not bool(
    os.environ.get("LANGKIT_NO_CUDA", False)
)
_device = "cuda" if _USE_CUDA else "cpu"

diagnostic_logger = getLogger(__name__)

try:
    import tensorflow as tf
except ImportError:
    tf = None


class CustomEncoder:
    def __init__(self, encoder: Callable):
        self.encode = encoder


class OnnxEncoder:
    def __init__(self, onnx_file_path: str = "all-MiniLM-L6-v2.onnx"):
        from os import environ
        from psutil import cpu_count

        environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))
        environ["OMP_WAIT_POLICY"] = "ACTIVE"

        import onnxruntime as ort
        from transformers import BertTokenizerFast

        _tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        _session = ort.InferenceSession(
            onnx_file_path, providers=["CPUExecutionProvider"]
        )

        def onnx_encode(texts):
            model_inputs = _tokenizer(texts, return_tensors="pt")
            inputs_onnx = {
                "input_ids": model_inputs["input_ids"].cpu().detach().numpy()
            }
            inputs_onnx["attention_mask"] = (
                model_inputs["attention_mask"].cpu().detach().numpy().astype(np.float32)
            )
            onnx_sequence = _session.run(None, inputs_onnx)
            embedding = OnnxEncoder.mean_pooling(
                model_output=onnx_sequence, attention_mask=inputs_onnx["attention_mask"]
            )
            return embedding[0]

        self.encode = onnx_encode

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        model_output = torch.from_numpy(model_output[0])
        token_embeddings = model_output
        attention_mask = torch.from_numpy(attention_mask)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask, input_mask_expanded, sum_mask


class Encoder:
    def __init__(
        self,
        transformer_name: Optional[str],
        custom_encoder: Optional[Callable[[List[str]], Any]],
        veto_cuda: bool = False,
        onnx: bool = False,
    ):
        """
        Args:
            transformer_name: The name of the transformer model to use. If None, a custom encoder must be provided.
                The name is expected to be a model name from the sentence_transformers library.
            custom_encoder: A custom encoder to use. If None, a transformer model must be provided.
                The custom encoder must be a callable that takes a list of strings and returns a list of embeddings.
        """
        if transformer_name and custom_encoder:
            raise ValueError(
                "Only one of transformer_name or encoder can be specified, not both."
            )
        if transformer_name is None and custom_encoder is None:
            raise ValueError(
                "One of transformer_name or custom_encoder must be specified, none was given."
            )
        elif onnx:
            with resources.path(__package__, "all-MiniLM-L6-v2.onnx") as path:
                onnx_transformer_name = str(path)
                transformer_model: Union[
                    CustomEncoder, OnnxEncoder, SentenceTransformer
                ] = OnnxEncoder(onnx_transformer_name)
            self.transformer_name = "onnx_encoder"
        elif custom_encoder:
            transformer_model = CustomEncoder(custom_encoder)
            self.transformer_name = "custom_encoder"
        elif transformer_name:
            device = _device if not veto_cuda else "cpu"
            transformer_model = SentenceTransformer(transformer_name, device=device)
            self.transformer_name = transformer_name
        self.transformer_model = transformer_model

    def encode(self, sentences: Union[List, str]) -> Union[Tensor, np.ndarray, List]:
        """
        Args:
            sentences: A list of sentences to encode. If a string is given, it is converted to a list with one element.
        Returns:
            A list of embeddings, one for each sentence in the input. The embeddings can be either a Tensor, a numpy array or a list.
            If the custom encoder returns a tensorflow tensor, it gets converted to a numpy array.
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        if isinstance(self.transformer_model, SentenceTransformer):
            embeddings = self.transformer_model.encode(
                sentences, convert_to_tensor=True
            )
        elif isinstance(self.transformer_model, (CustomEncoder, OnnxEncoder)):
            embeddings = self.transformer_model.encode(sentences)
        else:
            raise ValueError("Unknown encoder model type")
        if tf and isinstance(embeddings, tf.Tensor):
            embeddings = embeddings.numpy()
        return embeddings
