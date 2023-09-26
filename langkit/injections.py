from typing import Dict, List, Optional, Union
from whylogs.core.stubs import pd
from whylogs.experimental.core.udf_schema import register_dataset_udf
from . import prompt_column, lang_config
from sentence_transformers import SentenceTransformer
import requests
from io import BytesIO
import numpy as np
import faiss
from langkit.utils import _get_data_home
import os

_prompt = prompt_column
_index_embeddings = None
_transformer_model = None


def download_embeddings(url):
    response = requests.get(url)
    data = BytesIO(response.content)
    array = np.load(data)
    return array


def init(transformer_name: Optional[str] = None, version: Optional[str] = None):
    global _transformer_model
    global _index_embeddings
    if not transformer_name:
        transformer_name = "all-MiniLM-L6-v2"
    if not version:
        version = "v1"
    _transformer_model = SentenceTransformer(transformer_name)

    path = f"index_embeddings_{transformer_name}_harm_{version}.npy"
    embeddings_url = lang_config.injections_base_url + path
    embeddings_path = os.path.join(_get_data_home(), path)

    try:
        harm_embeddings = np.load(embeddings_path)
        save_embeddings = False
    except FileNotFoundError:
        try:
            harm_embeddings = download_embeddings(embeddings_url)

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
        _index_embeddings = faiss.deserialize_index(harm_embeddings)
        if save_embeddings:
            try:
                serialized_index = faiss.serialize_index(_index_embeddings)
                np.save(embeddings_path, serialized_index)
            except Exception as serialization_error:
                raise ValueError(
                    f"Injections - unable to serialize index to {embeddings_path}. Error: {serialization_error}"
                )
    except Exception as deserialization_error:
        raise ValueError(
            f"Injections - unable to deserialize index to {embeddings_path}. Error: {deserialization_error}"
        )


@register_dataset_udf([_prompt])
def injection(prompt: Union[Dict[str, List], pd.DataFrame]) -> Union[List, pd.Series]:
    global _transformer_model
    global _index_embeddings
    if _transformer_model is None:
        raise ValueError("Injections - transformer model not initialized")
    embeddings = _transformer_model.encode(prompt[_prompt])
    faiss.normalize_L2(embeddings)
    if _index_embeddings is None:
        raise ValueError("Injections - index embeddings not initialized")
    dists, _ = _index_embeddings.search(x=embeddings, k=1)
    return dists.flatten().tolist()


init()
