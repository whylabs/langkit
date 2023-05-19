from sentence_transformers import SentenceTransformer


def load_model(transformer_name: str):
    return SentenceTransformer(transformer_name)
