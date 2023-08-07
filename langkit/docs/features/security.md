# Security and Privacy

Monitoring for security and privacy in Language Model (LLM) applications helps ensuring the protection of user data and preventing malicious activities. Several approaches can be employed to strengthen the security and privacy measures within LLM systems.

One approach is to measure text similarity between prompts and responses against known examples of jailbreak attempts, prompt injections, and LLM refusals of service. By comparing the embeddings generated from the text, potential security vulnerabilities and unauthorized access attempts can be identified. This helps in mitigating risks and contributes to the LLM operation within secure boundaries. In langkit, text similarity calculation between prompts/responses and known examples of jailbreak attempts, prompt injections, and LLM refusals of service can be done through the [themes](../modules.md#themes) module.

Having a prompt injection classifier in place further enhances the security of LLM applications. By detecting and preventing prompt injection attacks, where malicious code or unintended instructions are injected into the prompt, the system can maintain its integrity and protect against unauthorized actions or data leaks. In langkit, prompt injection detection metrics can be computed through the [injections](../modules.md#injections) module.

Another important aspect of security and privacy monitoring involves checking prompts and responses against regex patterns designed to detect sensitive information. These patterns can help identify and flag data such as credit card numbers, telephone numbers, or other types of personally identifiable information (PII). In langkit, regex pattern matching against pattern groups can be done through the [regexes](../modules.md#regexes) module.

## Related Modules

- [themes](../modules.md#themes)
- [injections](../modules.md#injections)
