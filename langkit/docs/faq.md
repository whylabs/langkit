# Frequently Asked Questions

**Q**: Does Langkit integrate with which Large Language Models?

**A**: Most of Langkit's metrics require only the presence of one of two (or both) specific columns: a `prompt` column and a `response` column. You can use any LLM you wish with Langkit, as long as you provide the prompt and response columns. There are, however, two modules that require additional LLM calls to calculate metrics: `response_hallucination` and `proactive_injection_detection`. For these two modules, only OpenAI models are supported through `langkit.openai`'s `OpenAILegacy`, `OpenAIDefault`, and `OpenAIGPT4`. Azure-hosted OpenAI models are also supported through `OpenAIAzure`.

---

**Q**: Can I choose individual metrics in LangKit?

**A**: The finest granularity with which you can enable metrics is by metric namespaces. For example, you can import the `textstat` namespace, which will calculate all the text quality metrics (automated_readability_index,flesch_kincaid_grade, etc.) defined in that namespace. You can't, for example, pick the single metric `flesch_kincaid_grade`.

---

**Q**: Do I need to import each namespace I want to use manually?

**A**: Langkit provides some groups that will import multiple namespaces for you. Calling `llm_metrics`, for example, will automatically initialize a group of metrics usually used in conjunction for LLM applications. The `light_metrics` will initialize metrics that don't consume high computational resources and don't require big-sized artifacts to load. As its name suggests, the `all_metrics` group will initialize all available metric namespaces.

You can check the list of metric namespaces in each group in [this table](modules.md#modules-list).

---

**Q**: Can I use another encoder for the embedding similarity calculation?

**A**: You can define your own embeddings function to calculate the embeddings, as shown [in this example](../examples/Custom_Encoder.ipynb).

Additionally, as long as the model exists within the `sentence-transformers` library, you can define another model used for the embedding calculation through `LangkitConfig`'s `transformer_name` parameter.

---

**Q**: Can I use my own set of theme groups in the `themes` module?

**A**: Yes. You simply need to call `themes.init(theme_json=my_custom_themes)`, where `my_custom_themes` is your JSON formatted string.
