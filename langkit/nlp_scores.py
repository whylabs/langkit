from copy import deepcopy
from typing import List, Optional, Set
from whylogs.experimental.core.udf_schema import register_dataset_udf
import evaluate
from langkit import LangKitConfig, lang_config, response_column
from logging import getLogger


_corpus: str = lang_config.reference_corpus
_scores: List[str] = lang_config.nlp_scores
_rouge_type: str = lang_config.rouge_type


diagnostic_logger = getLogger(__name__)


_bleu_registered = False
_rouge_registered = False
_meteor_registered = False


def _register_score_udfs():
    global _bleu_registered, _rouge_registered, _meteor_registered

    if _corpus:
        for score in _scores:
            if "bleu" in score and not _bleu_registered:
                bleu = evaluate.load("bleu")
                _bleu_registered = True

                @register_dataset_udf(
                    [response_column],
                    udf_name=f"{response_column}.bleu_score",
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

            if "rouge" in score and not _rouge_registered:
                rouge = evaluate.load("rouge")
                _rouge_registered = True

                @register_dataset_udf(
                    [response_column],
                    udf_name=f"{response_column}.rouge_score",
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

            if "meteor" in score and not _meteor_registered:
                meteor = evaluate.load("meteor")
                _meteor_registered = True

                @register_dataset_udf(
                    [response_column],
                    udf_name=f"{response_column}.meteor_score",
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
    corpus: Optional[str] = None,
    scores: Set[str] = set(),
    rouge_type: str = "",
    config: Optional[LangKitConfig] = None,
):
    config = config or deepcopy(lang_config)
    global _corpus
    global _scores
    global _rouge_type
    _corpus = corpus or config.reference_corpus
    _scores = list(scores or config.nlp_scores)
    _rouge_type = rouge_type or config.rouge_type

    _register_score_udfs()


init()
