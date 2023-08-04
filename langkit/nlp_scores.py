from whylogs.experimental.core.udf_schema import register_dataset_udf
import evaluate
from . import LangKitConfig
from logging import getLogger

lang_config = LangKitConfig()
_corpus = lang_config.reference_corpus
_scores = lang_config.nlp_scores
_rouge_type = "rouge1"

diagnostic_logger = getLogger(__name__)


def register_score_udfs():
    if _corpus:
        for score in _scores:
            if "bleu" in score:
                bleu = evaluate.load("bleu")

                @register_dataset_udf([lang_config.response_column])
                def bleu_score(text):
                    result = []
                    for response in text[lang_config.response_column]:
                        result.append(
                            bleu.compute(predictions=[response], references=[_corpus])[
                                "bleu"
                            ]
                        )
                    return result

            if "rouge" in score:
                rouge = evaluate.load("rouge")

                @register_dataset_udf([lang_config.response_column])
                def rouge_score(text):
                    result = []
                    for response in text[lang_config.response_column]:
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

                @register_dataset_udf([lang_config.response_column])
                def meteor_score(text):
                    result = []
                    for response in text[lang_config.response_column]:
                        result.append(
                            meteor.compute(predictions=[text], references=[_corpus])[
                                "meteor"
                            ]
                        )
                    return result

    else:
        diagnostic_logger.warning(
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

    register_score_udfs()


init()
