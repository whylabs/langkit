import os
from functools import partial
from logging import getLogger
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from langkit.config import LANGKIT_CACHE
from langkit.core.metric import Metric, SingleMetric, SingleMetricResult
from langkit.metrics.util import LazyInit, retry
from langkit.transformer import sentence_transformer

logger = getLogger(__name__)

LANGKIT_INJECTIONS_CACHE: str = os.path.join(LANGKIT_CACHE, "injections")

__transformer_name = "all-MiniLM-L6-v2"
__version = "v2"
__injections_base_url = "https://whylabs-public.s3.us-west-2.amazonaws.com/langkit/data/injections/"
__embeddings = LazyInit(lambda: __load_embeddings())


@retry(max_attempts=3, wait_seconds=1)
def __download_parquet(url: str) -> pd.DataFrame:
    return pd.read_parquet(url)


def __cache_embeddings(harm_embeddings: pd.DataFrame, embeddings_path: str, filename: str) -> None:
    embeddings_path_local: str = os.path.join(embeddings_path, filename)
    try:
        os.makedirs(embeddings_path, exist_ok=True)
        harm_embeddings.to_parquet(embeddings_path_local)
    except Exception as serialization_error:
        logger.warning(f"Injections - unable to serialize embeddings to {embeddings_path_local}. Error: {serialization_error}")


def __download_embeddings(filename: str) -> pd.DataFrame:
    embeddings_path_remote: str = __injections_base_url + filename
    embeddings_path_local: str = os.path.join(LANGKIT_INJECTIONS_CACHE, filename)
    try:
        harm_embeddings = pd.read_parquet(embeddings_path_local)
        return harm_embeddings
    except FileNotFoundError:
        harm_embeddings = __download_parquet(embeddings_path_remote)
        __cache_embeddings(harm_embeddings, LANGKIT_INJECTIONS_CACHE, filename)
        return harm_embeddings
    except Exception as load_error:
        raise ValueError(f"Injections - unable to load embeddings from {embeddings_path_local}. Error: {load_error}")


def __process_embeddings(harm_embeddings: pd.DataFrame) -> "np.ndarray[Any, Any]":
    try:
        embeddings: Sequence[npt.ArrayLike] = harm_embeddings["sentence_embedding"].values  # type: ignore[reportUnknownMemberType]
        np_embeddings: "np.ndarray[Any, Any]" = np.stack(embeddings).astype(np.float32)
        embeddings_norm = np_embeddings / np.linalg.norm(np_embeddings, axis=1, keepdims=True)
        return embeddings_norm
    except Exception as e:
        raise ValueError(f"Injections - unable to process embeddings. Error: {e}")


def __load_embeddings() -> "np.ndarray[Any, Any]":
    filename = f"embeddings_{__transformer_name}_harm_{__version}.parquet"
    harm_embeddings = __download_embeddings(filename)
    embeddings_norm = __process_embeddings(harm_embeddings)
    return embeddings_norm


def injections_metric(column_name: str) -> Metric:
    def init():
        sentence_transformer.value(__transformer_name)
        __embeddings.value

    def udf(text: pd.DataFrame) -> SingleMetricResult:
        if column_name not in text.columns:
            raise ValueError(f"Injections: Column {column_name} not found in input dataframe")
        _embeddings = __embeddings.value
        _transformer = sentence_transformer.value(__transformer_name)
        target_embeddings: npt.NDArray[np.float32] = _transformer.encode(text[column_name])  # type: ignore[reportUnknownMemberType]
        target_norms = target_embeddings / np.linalg.norm(target_embeddings, axis=1, keepdims=True)
        cosine_similarities = np.dot(_embeddings, target_norms.T)
        max_similarities = np.max(cosine_similarities, axis=0)  # type: ignore[reportUnknownMemberType]
        max_indices = np.argmax(cosine_similarities, axis=0)
        metrics = [float(score) for _, score in zip(max_indices, max_similarities)]
        return SingleMetricResult(metrics=metrics)

    return SingleMetric(name=f"{column_name}.injections", input_name=column_name, evaluate=udf, init=init)


prompt_injections_module = partial(injections_metric, "prompt")
