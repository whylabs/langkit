LangKit is an open-source text metrics toolkit for monitoring language models. It offers an array of methods for extracting relevant signals from the input and/or output text, which are compatible with the open-source data logging library [whylogs](https://whylogs.readthedocs.io/en/latest).

## Table of Contents 📖

- [Motivation](#motivation)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)

## Motivation 🎯

Productionizing language models, including LLMs, comes with a range of risks due to the infinite amount of input combinations, which can elicit an infinite amount of outputs. The unstructured nature of text poses a challenge in the ML observability space - a challenge worth solving, since the lack of visibility on the model's behavior can have serious consequences.

## Features 🛠️

The currently supported metrics include:

- [Text Quality](https://github.com/whylabs/langkit/blob/main/langkit/docs/features/quality.md)
  - readability score
  - complexity and grade scores
- [Text Relevance](https://github.com/whylabs/langkit/blob/main/langkit/docs/features/relevance.md)
  - Similarity scores between prompt/responses
  - Similarity scores against user-defined themes
- [Security and Privacy](https://github.com/whylabs/langkit/blob/main/langkit/docs/features/security.md)
  - patterns - count of strings matching a user-defined regex pattern group
  - jailbreaks - similarity scores with respect to known jailbreak attempts
  - prompt injection - similarity scores with respect to known prompt injection attacks
  - refusals - similarity scores with respect to known LLM refusal of service responses
- [Sentiment and Toxicity](https://github.com/whylabs/langkit/blob/main/langkit/docs/features/sentiment.md)
  - sentiment analysis
  - toxicity analysis

## Installation 💻

To install LangKit, use the Python Package Index (PyPI) as follows:
```
pip install langkit
```

## Usage 🚀
LangKit modules contain UDFs that automatically wire into the collection of UDFs on String features provided by whylogs by default. All we have to do is import the LangKit modules and then instantiate a custom schema as shown in the example below.

```python
import whylogs as why
from whylogs.experimental.core.udf_schema import udf_schema
from langkit import sentiment
from langkit import textstat

results = why.log({"prompt": "Hello!", "response": "World!"}, schema=udf_schema())
```
The code above will produce a set of metrics comprised of the default whylogs metrics for text features and all the metrics defined in the imported modules.

More examples are available [here](https://github.com/whylabs/langkit/tree/main/langkit/examples).

## Modules 📦
You can have more information about the different modules and their metrics [here](https://github.com/whylabs/langkit/blob/main/langkit/docs/modules.md).