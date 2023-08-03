from whylogs.experimental.core.udf_schema import register_dataset_udf
import evaluate
from . import LangKitConfig
from logging import getLogger

lang_config = LangKitConfig()
_response = lang_config.response_column
_corpus = [lang_config.reference_corpus]
_scores = lang_config.nlp_scores
_rouge_type = "rouge1"

diagnostic_logger = getLogger(__name__)


def register_score_udfs():
    if _corpus:
        for score in _scores:
            if "bleu" in score:
                bleu = evaluate.load("bleu")

                @register_dataset_udf(
                    [_response],
                    namespace='response.performance'
                    )
                def bleu_score(cols) -> list:
                    series_results = []
                    for text in cols[_response]:
                        try:
                            score = bleu.compute(
                                        predictions=[text],
                                        references=_corpus
                                    )["bleu"]
                            series_results.append(score)
                        except Exception as e:
                            diagnostic_logger.warn(
                                f"Exception in bleu score udf: {e}"
                            )
                            series_results.append(None)
                    return series_results
                    

            if "rouge" in score:
                rouge = evaluate.load("rouge")

                @register_dataset_udf(
                    [_response],
                    namespace='response.performance',
                    udf_name=f"{_rouge_type}_score"
                    )
                def rouge_score(cols) -> list:
                    series_results = []
                    for text in cols[_response]:
                        try:
                            score = rouge.compute(
                                        predictions=[text],
                                        references=_corpus,
                                        rouge_types=[_rouge_type],
                                    )[_rouge_type]
                            series_results.append(score)
                        except Exception as e:
                            diagnostic_logger.warn(
                                f"Exception in rouge score udf: {e}"
                            )
                            series_results.append(None)
                    return series_results

            if "meteor" in score:
                meteor = evaluate.load("meteor")

                @register_dataset_udf(
                    [_response],
                    namespace='response.performance'
                    )
                def meteor_score(cols) -> list:
                    series_results = []
                    for text in cols[_response]:
                        try:
                            score = meteor.compute(
                                        predictions=[text],
                                        references=_corpus
                                    )["meteor"]
                            series_results.append(score)
                        except Exception as e:
                            diagnostic_logger.warn(
                                f"Exception in meteor score udf: {e}"
                            )
                            series_results.append(None)
                    return series_results

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
