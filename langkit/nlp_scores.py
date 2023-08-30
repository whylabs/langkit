from whylogs.experimental.core.udf_schema import register_dataset_udf
import evaluate
from . import lang_config, response_column
from logging import getLogger


_corpus = lang_config.reference_corpus
_scores = lang_config.nlp_scores
_rouge_type = "rouge1"

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


def init(corpus=None, scores=[], rouge_type=None):
    global _corpus
    global _scores
    global _rouge_type
    if corpus:
        _corpus = corpus
    if scores:
        _scores = scores
    if rouge_type:
        _rouge_type = rouge_type

    _register_score_udfs()


init()
