# Changes

## Glossary

- Feature Modules: Stuff like `toxicity`,`textstat`, `sentiment`, etc.
- Feature Presets: A set of feature modules. Possible presets: `light_features`,`llm_features`,`security`,`leakage`, etc.

## No auto-init

Feature modules are no longer auto-initialized. This means that you need to call init for every module you want to use. This is a breaking change.

Example:

```python
from langkit import toxicity, extract

schema = toxicity.init()

enhanced_df = extract(df, schema=schema)
```

## Every module's init returns a schema

Every module's init function now returns a schema. The schema should contain only the UDFs that the respective module provides (no global udf registry). This is a breaking change.

Example:

```python
from langkit import toxicity, sentiment, extract

sentiment_schema = sentiment.init()
toxicity_schema = toxicity.init() 

enhanced_df = extract(df, schema=schema)
```

## Schema Mergeability

Schemas are mergeable, and that should be the way to build Feature Presets (built-in or custom)

```python
from langkit.presets import llm_features
llm_schema = llm_features.init(config: LangkitConfig)
enhanced_df = extract(df, schema=my_schema)
```

```python
from langkit import toxicity, sentiment, extract

sentiment_schema = sentiment.init()
toxicity_schema = toxicity.init() 

my_schema = sentiment_schema.merge(toxicity_schema)

enhanced_df = extract(df, schema=my_schema)
```

## Config centralization

Every feature module should be fully customizable through a `LangkitConfig` object.

The reason is that, when initializing a feature preset, we need to be able to pass a single config object to every module.

Example:

```python
from langkit import textstat, regexes, extract, lang_config
from langkit.presets import light_features

textstat_schema = textstat.init(lang_config)
regexes_schema = regexes.init(lang_config) 

my_schema = textstat_schema.merge(regexes_schema)

light_schema = light_features.init(config=lang_config)

#light_schema should equal my_schema
#light_features should have the same customization possibilites as my_schema
```

## Multilingual Support

Should we include multilingual support in v0.1?

## Openai dependency

Currently, we support `openai` v0 and v1. Should we clean the code and support only openai v1 in v0.1.0?
