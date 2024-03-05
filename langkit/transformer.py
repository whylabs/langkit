from sentence_transformers import SentenceTransformer
from typing import Optional, Callable, Union, List, Any
from torch import Tensor
import numpy as np
from functools import lru_cache
import os
import torch

_USE_CUDA = torch.cuda.is_available() and not bool(
    os.environ.get("LANGKIT_NO_CUDA", False)
)
_device = "cuda" if _USE_CUDA else "cpu"


@lru_cache(maxsize=None)
def _get_sentence_transformer(model_name: str, veto_cuda=False) -> SentenceTransformer:
    device = _device if not veto_cuda else "cpu"
    return SentenceTransformer(model_name, device=device)


try:
    import tensorflow as tf
except ImportError:
    tf = None


class CustomEncoder:
    def __init__(self, encoder: Callable):
        self.encode = encoder


class Encoder:
    def __init__(
        self,
        transformer_name: Optional[str],
        custom_encoder: Optional[Callable[[List[str]], Any]],
        veto_cuda: bool = False,
    ):
        """
        Args:
            transformer_name: The name of the transformer model to use. If None, a custom encoder must be provided.
                The name is expected to be a model name from the sentence_transformers library.
            custom_encoder: A custom encoder to use. If None, a transformer model must be provided.
                The custom encoder must be a callable that takes a list of strings and returns a list of embeddings.
        """
        self.veto_cuda = veto_cuda

        if transformer_name and custom_encoder:
            raise ValueError(
                "Only one of transformer_name or encoder can be specified, not both."
            )
        if transformer_name is None and custom_encoder is None:
            raise ValueError(
                "One of transformer_name or custom_encoder must be specified, none was given."
            )
        if custom_encoder:
            self.transformer_name = None
            self.custom_encoder: Optional[CustomEncoder] = CustomEncoder(custom_encoder)
        if transformer_name:
            self.transformer_name = transformer_name
            self.custom_encoder = None

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
        if self.custom_encoder:
            embeddings = self.custom_encoder.encode(sentences)
        elif self.transformer_name:
            transformer_model = _get_sentence_transformer(
                self.transformer_name, self.veto_cuda
            )
            embeddings = transformer_model.encode(sentences, convert_to_tensor=True)
        else:
            raise ValueError("Unknown encoder model type")
        if tf and isinstance(embeddings, tf.Tensor):
            embeddings = embeddings.numpy()
        return embeddings
