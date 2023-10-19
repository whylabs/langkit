from typing import List, Optional
from langkit.eval_llm.base_llm import LLMConfig
from langkit.eval_llm import get_llm
from langkit.eval_llm.base_llm import BaseLLM, ChatLog
from logging import getLogger
from whylogs.experimental.core.udf_schema import register_dataset_udf
from langkit import LangKitConfig, lang_config, prompt_column, response_column
from nltk.tokenize import sent_tokenize
from langkit.openai.openai import LLMInvocationParams, Conversation
from langkit.transformer import Encoder
from sentence_transformers import util

_prompt = prompt_column
_response = response_column

diagnostic_logger = getLogger(__name__)

embeddings_encoder = Encoder(lang_config.transformer_name, custom_encoder=None)


class ConsistencyCheker:
    def __init__(self, llm: LLMInvocationParams, num_samples, embeddings_encoder):
        self.num_samples = num_samples
        self.llm = llm
        sample_generator = llm.copy()
        sample_generator.temperature = 1
        self.sample_generator_llm = sample_generator
        consistency_checker_llm = llm.copy()
        consistency_checker_llm.temperature = 0
        consistency_checker_llm.max_tokens = 10
        self.consistency_checker_llm = consistency_checker_llm
        self.embeddings_encoder = embeddings_encoder

    def convert_score(self, result):
        categories = {
            "major inaccurate": 1.0,
            "minor inaccurate": 0.5,
            "accurate": 0.0,
        }
        score = categories.get(result.lower(), None)
        if not score:
            diagnostic_logger.info(
                f"Invalid result from consistency checker: {result}. Valid results are {categories.keys()}"
            )
        return score

    def get_samples(self, prompt):
        samples = []
        for _ in range(self.num_samples):
            response: ChatLog = Conversation(self.sample_generator_llm).send_prompt(
                prompt
            )
            samples.append(response)
        return samples

    def sentence_similarity(self, response_sentence, samples_list, embeddings_encoder):
        sample_similarities = []
        for sample in samples_list:
            sample_sentences = sent_tokenize(sample)
            sentence_similarities = [
                util.pytorch_cos_sim(
                    embeddings_encoder.encode(response_sentence),
                    embeddings_encoder.encode(sample_sentence),
                ).item()
                for sample_sentence in sample_sentences
            ]
        sample_similarities.append(max(sentence_similarities))
        sentence_score = 1 - sum(sample_similarities) / len(sample_similarities)
        return sentence_score

    def semantic_similarity(self, response, additional_samples):
        response_sentences = sent_tokenize(response)
        samples_list = [sample.response for sample in additional_samples]
        response_similarities = [
            self.sentence_similarity(sentence, samples_list, self.embeddings_encoder)
            for sentence in response_sentences
        ]
        final_score = sum(response_similarities) / len(response_similarities)
        return final_score

    def llm_consistency_check(self, response, additional_samples):
        llm_scores = []
        for sample in additional_samples:
            prompt = f"Context: {sample.response}\n\nPassage: {response}\n\nIs the passage supported by the context above? Answer between: Accurate, Minor Inaccurate, Major Inaccurate "
            result: ChatLog = Conversation(self.consistency_checker_llm).send_prompt(
                prompt
            )
            llm_score = self.convert_score(result.response)
            llm_scores.append(llm_score)
        final_score = sum(llm_scores) / len(llm_scores)
        return final_score

    def consistency_check(self, response, additional_samples: List[ChatLog]):
        llm_score = self.llm_consistency_check(response, additional_samples)
        semantic_score = self.semantic_similarity(response, additional_samples)
        return (llm_score + semantic_score) / 2


checker: Optional[ConsistencyCheker] = None


def init(llm: LLMInvocationParams, num_samples=1):
    global checker
    checker = ConsistencyCheker(llm, num_samples, embeddings_encoder)


@register_dataset_udf([_prompt, _response], f"{_response}.consistency")
def response_consistency(text):
    series_result = []
    for prompt, response in zip(text[_prompt], text[_response]):
        additional_samples = checker.get_samples(prompt)
        result = checker.consistency_check(response, additional_samples)
        series_result.append(result)
    return series_result
