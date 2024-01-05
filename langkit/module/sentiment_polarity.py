from functools import partial
from typing import Any, Dict, List, Union

import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from langkit.module.module import UdfInput, UdfSchemaArgs
from whylogs.experimental.core.udf_schema import NO_FI_RESOLVER, UdfSpec


def __sentiment_polarity_module(column_name: str, lexicon: str = "vader_lexicon") -> UdfSchemaArgs:
    # TODO Does this have built in idempotency?
    nltk.download(lexicon)  # type: ignore[reportUnknownMemberType]
    analyzer = SentimentIntensityAnalyzer()

    def udf(text: Union[pd.DataFrame, Dict[str, List[Any]]]) -> Any:
        return [analyzer.polarity_scores(t)["compound"] for t in UdfInput(text).iter_column_rows(column_name)]  # type: ignore[reportUnknownMemberType]

    textstat_udf = UdfSpec(
        column_names=[column_name],
        udfs={f"{column_name}.sentiment_polarity": udf},
    )

    schema = UdfSchemaArgs(
        types={column_name: str},
        resolvers=NO_FI_RESOLVER,
        udf_specs=[textstat_udf],
    )

    return schema


promp_sentiment_polarity = partial(__sentiment_polarity_module, "prompt")
response_sentiment_polarity = partial(__sentiment_polarity_module, "response")
promp_response_sentiment_polarity = [promp_sentiment_polarity, response_sentiment_polarity]
