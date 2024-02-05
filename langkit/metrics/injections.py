import os
from functools import partial
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from langkit.core.metric import Metric, SingleMetric, SingleMetricResult
from langkit.metrics.util import LazyInit, get_data_home

__transformer_name = "all-MiniLM-L6-v2"
__version = "v2"
__injections_base_url = "https://whylabs-public.s3.us-west-2.amazonaws.com/langkit/data/injections/"
__embeddings = LazyInit(lambda: __load_embeddings())
__transformer = LazyInit(
    lambda: SentenceTransformer(__transformer_name, device="cuda" if torch.cuda.is_available() else "cpu")
)

def __load_embeddings() -> "np.ndarray[Any, Any]":
    filename = f"embeddings_{__transformer_name}_harm_{__version}.parquet"
    embeddings_url: str = __injections_base_url+filename
    embeddings_path: str = os.path.join(get_data_home(),filename)
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
        embeddings: Sequence[npt.ArrayLike] = harm_embeddings["sentence_embedding"].values  # type: ignore[reportUnknownMemberType]
        np_embeddings: "np.ndarray[Any, Any]" = np.stack(embeddings).astype(
            np.float32
        )
        embeddings_norm = np_embeddings / np.linalg.norm(
            np_embeddings, axis=1, keepdims=True
        )

        if save_embeddings:
            try:
                harm_embeddings.to_parquet(embeddings_path)
            except Exception as serialization_error:
                raise ValueError(
                    f"Injections - unable to serialize index to {embeddings_path}. Error: {serialization_error}"
                )
        return embeddings_norm
    except Exception as deserialization_error:
        raise ValueError(
            f"Injections - unable to deserialize index to {embeddings_path}. Error: {deserialization_error}"
        )

def injections_metric(column_name:str) -> Metric:
    def init():
        global _filename
        __transformer.value

    def udf(text: pd.DataFrame) -> SingleMetricResult:
        if column_name not in text.columns:
            raise ValueError(f"Injections: Column {column_name} not found in input dataframe")
        _embeddings = __embeddings.value
        _transformer = __transformer.value
        target_embeddings: npt.NDArray[np.float32] = _transformer.encode(text[column_name]) # type: ignore[reportUnknownMemberType]
        target_norms = target_embeddings / np.linalg.norm(
            target_embeddings, axis=1, keepdims=True
        )
        cosine_similarities = np.dot(_embeddings, target_norms.T)
        max_similarities = np.max(cosine_similarities, axis=0) # type: ignore[reportUnknownMemberType]
        max_indices = np.argmax(cosine_similarities, axis=0)
        metrics = [float(score) for _, score in zip(max_indices, max_similarities)]
        return SingleMetricResult(metrics=metrics)

    return SingleMetric(name=f"{column_name}.injections", input_name=column_name, evaluate=udf, init=init)

prompt_injections_module = partial(injections_metric, "prompt")
