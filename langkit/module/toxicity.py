# pyright: reportUnknownMemberType=none
# pyright: reportUnknownVariableType=none
import os
from functools import partial
from typing import Any, Dict, List, Optional, Union, cast

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TextClassificationPipeline,
)

from langkit.module.module import UdfInput, UdfSchemaArgs
from whylogs.experimental.core.udf_schema import NO_FI_RESOLVER, UdfSpec


def __toxicity(pipeline: TextClassificationPipeline, max_length: int, text: List[str]) -> float:
    # TODO lots of error handling here
    results = pipeline(text, truncation=True, max_length=max_length)
    return [result["score"] if result["label"] == "toxic" else 1 - result["score"] for result in results]


__model: Optional[PreTrainedTokenizerBase] = None
__tokenizer: Optional[PreTrainedTokenizerBase] = None


def __toxicity_module(column_name: str) -> UdfSchemaArgs:
    global __model, __tokenizer
    model_path = "martin-ha/toxic-comment-model"

    if __model is None:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        __model = model
    else:
        model = __model

    if __tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        __tokenizer = tokenizer
    else:
        tokenizer = __tokenizer

    use_cuda = torch.cuda.is_available() and not bool(os.environ.get("LANGKIT_NO_CUDA", False))
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0 if use_cuda else -1)  # type: ignore[reportUnknownArgumentType]
    # TODO test/error handling
    max_length = cast(int, tokenizer.model_max_length)

    def _udf(column_name: str, text: Union[pd.DataFrame, Dict[str, List[Any]]]) -> Any:
        col = list(UdfInput(text).iter_column(column_name))
        # TODO filter out only the non-strings
        return __toxicity(pipeline, max_length, col)

    udf = partial(_udf, column_name)

    textstat_udf = UdfSpec(
        column_names=[column_name],
        udfs={f"{column_name}.toxicity": udf},
    )

    schema = UdfSchemaArgs(
        types={column_name: str},
        resolvers=NO_FI_RESOLVER,
        udf_specs=[textstat_udf],
    )

    return schema


prompt_toxicity = partial(__toxicity_module, "prompt")
response_toxicity = partial(__toxicity_module, "response")
# TODO make these lists readonly
prompt_response_toxicity = [prompt_toxicity, response_toxicity]
