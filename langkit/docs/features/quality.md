# Text Quality

Text quality metrics, such as readability, complexity and grade level, can provide important insights into the quality and appropriateness of generated responses. By monitoring these metrics, we can ensure that the Language Model (LLM) outputs are clear, concise, and suitable for the intended audience.

Assessing text complexity and grade level assists in tailoring the generated content to the target audience. By considering factors such as sentence structure, vocabulary choice, and domain-specific requirements, we can ensure that the LLM produces responses that align with the intended reading level and professional context. Additionally, incorporating metrics such as syllable count, word count, and character count allows us to closely monitor the length and composition of the generated text. By setting appropriate limits and guidelines, we can ensure that the responses remain concise, focused, and easily digestible for users.

In **langkit**, we can compute text quality metrics through the [textstat](../modules.md#text-statistics) module, which uses the [textstat](https://github.com/textstat/textstat) library to compute several different text quality metrics.

## Related Modules

- [textstat](../modules.md#text-statistics)
