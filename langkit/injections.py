from copy import deepcopy
from typing import Dict, List, Optional, Union
from whylogs.experimental.core.udf_schema import register_dataset_udf
from langkit import LangKitConfig, lang_config, prompt_column
from sentence_transformers import SentenceTransformer
import numpy as np
from langkit.utils import _get_data_home
import os
import torch
import pandas as pd

_prompt = prompt_column
_transformer_model = None
_embeddings_norm = None

_USE_CUDA = torch.cuda.is_available() and not bool(
    os.environ.get("LANGKIT_NO_CUDA", False)
)
_device = "cuda" if _USE_CUDA else "cpu"


def init(
    transformer_name: Optional[str] = None,
    version: Optional[str] = "v2",
    config: Optional[LangKitConfig] = None,
):
    config = config or deepcopy(lang_config)

    global _transformer_model
    global _embeddings_norm
    if not transformer_name:
        transformer_name = "all-MiniLM-L6-v2"
    _transformer_model = SentenceTransformer(transformer_name, device=_device)
    path = f"embeddings_{transformer_name}_harm_{version}.parquet"
    embeddings_url = config.injections_base_url + path
    embeddings_path = os.path.join(_get_data_home(), path)

    try:
        harm_embeddings = pd.read_parquet(embeddings_path)
        save_embeddings = False
    except FileNotFoundError:
        try:
            harm_embeddings = pd.read_parquet(embeddings_url)

        except Exception as download_error:
            raise ValueError(
                f"Injections - unable to download embeddings from {embeddings_url}. Error: {download_error}"
            )
        save_embeddings = True
    except Exception as load_error:
        raise ValueError(
            f"Injections - unable to load embeddings from {embeddings_path}. Error: {load_error}"
        )

    try:
        array_list = [np.array(x) for x in harm_embeddings["sentence_embedding"].values]
        np_embeddings = np.stack(array_list).astype(np.float32)

        _embeddings_norm = np_embeddings / np.linalg.norm(
            np_embeddings, axis=1, keepdims=True
        )

        if save_embeddings:
            try:
                harm_embeddings.to_parquet(embeddings_path)
            except Exception as serialization_error:
                raise ValueError(
                    f"Injections - unable to serialize index to {embeddings_path}. Error: {serialization_error}"
                )
    except Exception as deserialization_error:
        raise ValueError(
            f"Injections - unable to deserialize index to {embeddings_path}. Error: {deserialization_error}"
        )


@register_dataset_udf([_prompt], f"{_prompt}.injection")
def injection(prompt: Union[Dict[str, List], pd.DataFrame]) -> List:
    global _transformer_model
    global _embeddings_norm

    if _transformer_model is None:
        raise ValueError("Injections - transformer model not initialized")
    if _embeddings_norm is None:
        raise ValueError("Injections - embeddings not initialized")
    target_embeddings = _transformer_model.encode(prompt[_prompt])
    target_norms = target_embeddings / np.linalg.norm(
        target_embeddings, axis=1, keepdims=True
    )
    cosine_similarities = np.dot(_embeddings_norm, target_norms.T)
    max_similarities = np.max(cosine_similarities, axis=0)
    max_indices = np.argmax(cosine_similarities, axis=0)
    return [float(score) for _, score in zip(max_indices, max_similarities)]


init()
