from whylogs.experimental.core.metrics.udf_metric import register_metric_udf
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

                @register_metric_udf(col_name=lang_config.response_column)
                def bleu_score(text: str) -> float:
                    return bleu.compute(predictions=[text], references=[_corpus])[
                        "bleu"
                    ]

            if "rouge" in score:
                rouge = evaluate.load("rouge")

                @register_metric_udf(col_name=lang_config.response_column)
                def rouge_score(text: str) -> float:
                    return rouge.compute(
                        predictions=[text],
                        references=[_corpus],
                        rouge_types=[_rouge_type],
                    )[_rouge_type]

            if "meteor" in score:
                meteor = evaluate.load("meteor")

                @register_metric_udf(col_name=lang_config.response_column)
                def meteor_score(text: str) -> float:
                    return meteor.compute(predictions=[text], references=[_corpus])[
                        "meteor"
                    ]

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
