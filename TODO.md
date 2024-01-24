

# Release checklist
- Audit import paths, make sure everything is organized in a sensible fashion.
- Complete the lib export directory
- Make sure names are consistent. Module -> Metric most likely
- Add `pre_init` methods to all of the modules that can download models so we can force downloads upfront. These should probably be called
  from the library creator methods.
- Update the default topics in the topic metric so they're useful. Right now, you woud always want to specify your own.
- Document the metric names in the default metrics, and how to get them programatically
- Figure out what's wrong with the automatic datasetschema convertsion. There are warnings in the tests for
    - WARNING:whylogs.core.resolvers:Conflicting resolvers for counts metric in column 'response.upper_case_char_count' of type int
- Make sure frequent items metrics are never enabled on prompt/response/input col name
- pre-init for vader
```
[nltk_data] Downloading package vader_lexicon to /root/nltk_data...
[nltk_data]   Package vader_lexicon is already up-to-date!
[nltk_data] Downloading package vader_lexicon to /root/nltk_data...
[nltk_data]   Package vader_lexicon is already up-to-date!
```
- Fail if the validators is trying to validate a column that will never exist 
- Validate the return types of metrics. We should ensure that the return types are native python whenever possible. Numpy/tensors don't
  serialize which just turns into potential issues down the road, especially in reguard to multiprocessing.
    - Same goes for validators. Anytime they index into the data frame that we provide the output is going to be a numpy type, which is not
      serializable.
- Regexes are the odd man out with respect to config. I think we should get rid of the concept of a json file for regexes and treat it like
  any other metric. The version control and policy options should all be in one place: the platform/container.
- Validator structure might need to be more flexible, allowing for gte, ==, for example.
