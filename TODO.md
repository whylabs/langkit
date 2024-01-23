

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
