import torch
from sentence_transformers import SentenceTransformer

from langkit.metrics.util import DynamicLazyInit

sentence_transformer: DynamicLazyInit[str, SentenceTransformer] = DynamicLazyInit(
    lambda model_name: SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
)
