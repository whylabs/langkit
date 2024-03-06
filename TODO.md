

# Release checklist
- DONE Audit import paths, make sure everything is organized in a sensible fashion.
- Complete the lib export directory
- Make sure names are consistent. Module -> Metric most likely
- DONE Add `pre_init` methods to all of the modules that can download models so we can force downloads upfront. These should probably be called
  from the library creator methods.
- DONE Update the default topics in the topic metric so they're useful. Right now, you woud always want to specify your own.
- Document the metric names in the default metrics, and how to get them programatically
- maFigure out what's wrong with the automatic datasetschema convertsion. There are warnings in the tests for
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
- Make use of the input_name in metrics. It's left over for whylogs schema generation but nothing in the new api needs it. It could be used
  like the target_names in validators to narrow down the data frame that gets passed into the metric udf, then you would have to declare
  what you need and we would have more validation power.
- Add async versions of the hook apis. Adds some complexity interfacing with the asyncio loop.
- How do eegments fit into this? No segment notion in langkit atm and we can't automatically assume anything about segments in the
  whylogs_compat thing. I think we would need to force more manual whylogs setup for the advanced cases and avoid the auto stuff.
- Implement real multi metric conversion in the whylogs_compat file. Jamie said that whylogs actually can return multiple metrics at once.
- Add validation options for string based metrics like topic. Right now there are only validation creators for numeric things.
- Add multiple python version builds to the CI matrix. Things like `isinstance(foo, Union[..])` are in the code still.
- Figure out how to get the CI build working for 3.12. Numpy won't actually install there because of removed distutils.
