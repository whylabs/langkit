# Langkit
**Langkit** is an open-source text metrics toolkit for monitoring language models. It offers an array of methods for extracting relevant signals from the input and/or output text, which are compatible with the open-source data logging library [whylogs](https://whylogs.readthedocs.io/en/latest).

The generated profiles can be visualized and monitored in the [WhyLabs platform](https://whylabs.ai/) or they can be further analyzed by the user on their own accord.

## Motivation

Productionalizing language models, including LLMs, comes with a range of risks due to the infinite amount of input combinations, which can elicit an infinite amount of outputs. The unstructured nature of text poses a challenge in the ML observability space - a challenge worth solving, since the lack of visibility on the model's behavior can have serious consequences.

## Features

The currently supported metrics include:
- readability
- complexity
- grade
- sentiment
- similarity to a user-defined topic

## Installation

To install Langkit, use the Python Package Index (PyPI) as follows:
```bash
pip install langkit
```

## Usage

Langkit modules contain UDFs that automatically wire into the collection of UDFs on String features provided by whylogs by default. All we have to do is import the Langkit modules and then instantiate a custom schema as shown in the example below.

```python 
from whylogs.experimental.core.metrics.udf_metric import generate_udf_schema
from whylogs.core.resolvers import STANDARD_RESOLVER
from whylogs.core.schema import DatasetSchema, DeclarativeSchema
import whylogs as why
from langkit.sentiment import *
from langkit.textstat import *

def standard_udf_schema() -> DatasetSchema:
    return DeclarativeSchema(STANDARD_RESOLVER + generate_udf_schema())

schema=standard_udf_schema()

results = why.log(df, schema=schema)

```
The code above will produce a set of metrics comprised of the default whylogs metrics for text features and all the metrics defined in the imported modules.

More examples available [here](https://github.com/whylabs/LanguageToolkit/tree/main/examples)
