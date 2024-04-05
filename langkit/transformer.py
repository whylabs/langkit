from functools import lru_cache
from typing import Tuple

import torch
from sentence_transformers import SentenceTransformer

from langkit.metrics.embeddings_types import CachingEmbeddingEncoder, EmbeddingEncoder
from langkit.onnx_encoder import OnnxSentenceTransformer, TransformerModel


@lru_cache
def sentence_transformer(
    name_revision: Tuple[str, str] = ("all-MiniLM-L6-v2", "44eb4044493a3c34bc6d7faae1a71ec76665ebc6"),
) -> SentenceTransformer:
    """
    Returns a SentenceTransformer model instance.

    The intent of this function is to cache the SentenceTransformer instance to avoid
    multple instances being created all over langkit, and have a single place that
    can be used to change the transformer name for the metrics that default to the same one.
    """
    transformer_name, revision = name_revision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(transformer_name, revision=revision, device=device)


@lru_cache
def embedding_adapter() -> EmbeddingEncoder:
    return CachingEmbeddingEncoder(OnnxSentenceTransformer(TransformerModel.AllMiniLM))
