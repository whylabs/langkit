from typing import List

import torch
import torch.nn.functional as F

from langkit.metrics.embeddings_types import EmbeddingEncoder


def compute_embedding_similarity(encoder: EmbeddingEncoder, _in: List[str], _out: List[str]) -> torch.Tensor:
    in_encoded = torch.as_tensor(encoder.encode(_in))
    out_encoded = torch.as_tensor(encoder.encode(_out))
    print(f"computing similarities between {in_encoded.shape} and {out_encoded.shape}")
    sim = F.cosine_similarity(in_encoded, out_encoded, dim=1)

    print(f"computed similarity: {sim}")
    return sim
