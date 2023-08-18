from sentence_transformers import SentenceTransformer
from typing import Optional, Callable, Union, List, Any
from torch import Tensor
import numpy as np

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
                "One of transformer_name or encoder must be specified, none was given."
            )
        if custom_encoder:
            transformer_model = CustomEncoder(custom_encoder)
            self.transformer_name = "custom_encoder"
        if transformer_name:
            transformer_model = SentenceTransformer(transformer_name)
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
        elif isinstance(self.transformer_model, CustomEncoder):
            embeddings = self.transformer_model.encode(sentences)
        else:
            raise ValueError("Unknown encoder model type")
        if tf and isinstance(embeddings, tf.Tensor):
            embeddings = embeddings.numpy()
        return embeddings
