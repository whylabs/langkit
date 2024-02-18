from typing import List

import torch
import torch.nn.functional as F

from langkit.metrics.embeddings_types import TransformerEmbeddingAdapter


def compute_embedding_similarity(encoder: TransformerEmbeddingAdapter, _in: List[str], _out: List[str]) -> torch.Tensor:
    in_encoded = torch.as_tensor(encoder.encode(tuple(_in)))
    out_encoded = torch.as_tensor(encoder.encode(tuple(_out)))
    return F.cosine_similarity(in_encoded, out_encoded, dim=1)
