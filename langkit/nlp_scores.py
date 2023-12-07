from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Set
from whylogs.experimental.core.udf_schema import register_dataset_udf
import evaluate
from langkit import LangKitConfig, lang_config, response_column
from logging import getLogger
from langkit.whylogs.unreg import unregister_udfs


_corpus: Optional[str] = lang_config.reference_corpus
_scores: List[str] = lang_config.nlp_scores
_rouge_type: str = lang_config.rouge_type


diagnostic_logger = getLogger(__name__)

_initialized = False

_registered: Dict[str, Set[str]] = defaultdict(
    set
)  # _registered[schema_name] -> set of registered UDF names


def _register_score_udfs(schema_name: str):
    if not _initialized:
        init()
    global _registered
    unregister_udfs(_registered[schema_name], schema_name=schema_name)
    _registered[schema_name] = set()
    if _corpus:
        for score in _scores:
            if "bleu" in score:
                bleu = evaluate.load("bleu")
                _registered[schema_name].add(f"{response_column}.bleu_score")

                @register_dataset_udf(
                    [response_column],
                    udf_name=f"{response_column}.bleu_score",
                    schema_name=schema_name,
                )
                def bleu_score(text):
                    result = []
                    for response in text[response_column]:
                        result.append(
                            bleu.compute(predictions=[response], references=[_corpus])[
                                "bleu"
                            ]
                        )
                    return result

            if "rouge" in score:
                rouge = evaluate.load("rouge")
                _registered[schema_name].add(f"{response_column}.rouge_score")

                @register_dataset_udf(
                    [response_column],
                    udf_name=f"{response_column}.rouge_score",
                    schema_name=schema_name,
                )
                def rouge_score(text):
                    result = []
                    for response in text[response_column]:
                        result.append(
                            rouge.compute(
                                predictions=[text],
                                references=[_corpus],
                                rouge_types=[_rouge_type],
                            )[_rouge_type]
                        )
                    return result

            if "meteor" in score:
                meteor = evaluate.load("meteor")
                _registered[schema_name].add(f"{response_column}.meteor_score")

                @register_dataset_udf(
                    [response_column],
                    udf_name=f"{response_column}.meteor_score",
                    schema_name=schema_name,
                )
                def meteor_score(text):
                    result = []
                    for response in text[response_column]:
                        result.append(
                            meteor.compute(predictions=[text], references=[_corpus])[
                                "meteor"
                            ]
                        )
                    return result

    else:
        diagnostic_logger.info(
            "No reference corpus provided for NLP scores. Skipping NLP scores."
        )


def init(
    language: Optional[str] = None,
    corpus: Optional[str] = None,
    scores: Set[str] = set(),
    rouge_type: str = "",
    config: Optional[LangKitConfig] = None,
    schema_name: str = "",
):
    global _initialized
    _initialized = True
    config = config or deepcopy(lang_config)
    global _corpus
    global _scores
    global _rouge_type
    _corpus = corpus or config.reference_corpus
    _scores = list(scores or config.nlp_scores)
    _rouge_type = rouge_type or config.rouge_type

    _register_score_udfs(schema_name)
