import torch
import torch.nn.functional as F


def compute_embedding_similarity_encoded(in_encoded: torch.Tensor, out_encoded: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(in_encoded, out_encoded, dim=1)
