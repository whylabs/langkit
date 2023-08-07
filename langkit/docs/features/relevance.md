# Text Relevance

Text relevance plays a crucial role in the monitoring of Language Models (LLMs) by providing an objective measure of the similarity between different texts. It serves multiple use cases, including assessing the quality and appropriateness of LLM outputs and providing guardrails to ensure the generation of safe and desired responses.

One use case is computing similarity scores between embeddings generated from prompts and responses, enabling the evaluation of the relevance between them. This helps identify potential issues such as irrelevant or off-topic responses, ensuring that LLM outputs align closely with the intended context. In **langkit**, we can compute similarity scores between prompt and response pairs using the [input_output](../modules.md#inputoutput) module.

Another use case is calculating the similarity of prompts and responses against certain topics or known examples, such as jailbreaks or controversial subjects. By comparing the embeddings to these predefined themes, we can establish guardrails to detect potential dangerous or unwanted responses. The similarity scores serve as signals, alerting us to content that may require closer scrutiny or mitigation. In **langkit**, this can be done through the [themes](../modules.md#themes) module.

By leveraging text relevance as a monitoring metric for LLMs, we can not only evaluate the quality of generated responses but also establish guardrails to minimize the risk of generating inappropriate or harmful content. This approach enhances the performance, safety, and reliability of LLMs in various applications, providing a valuable tool for responsible AI development.

## Related Modules

- [input_output](../modules.md#inputoutput)
- [themes](../modules.md#themes)
- [topics](../modules.md#topics)
