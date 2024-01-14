# V2

## Questions

- Is pandas really the right interface for consuming/producing results in evaluation? It would be a pretty fundemental deendency that we
  could really never shake, but it is a familiar thing. It has implications for our own dependency modeling, who could consume us, how we
  serialize, etc.
- How can we make metrics and metric names more discoverable? Python's type system just can't do the cool stuff that TypeScript's can, and
  even if it could, a lot of people might just miss it while using non-ide environments like notebooks that don't have good type support.
  The `lib` pattern works pretty well for organizing the metrics that we create, but the dataframe that pops out the other end is still just
  string names that does't correspond to the functions in the lib pattern.
- Should we rethink the way that we name metrics now that we're free of whylogs? The problem with our current naming scheme is that it's
  impossible to know what the name is going to be programatically. The user has to just know that it's going to end up as foo.something
  maybe. And sometimes its even more complicated, ending up as patterns.<name_from_regex_json>.

  - One idea: Don't hard code the metric output names? Force them to pick one and use a default value that propagates down? At least that
    way there would always be some method to determine what this thing would end up as.
  - The problem with that is that creating groups of metrics becomes really weird. You can't have the current nice array groupings that
    just assume some composed name anymore, you would have to somehow assign names manually to all of the underlying metrics. Maybe we can
    get around names entirely? Need to see the cases where you would want to actually use the metric names at all. Maybe the
    Moderator/Blocker abstraction just deliver judgements for you based on a config and you never even care about their exact names.

- In the prototype, the pattern for making metrics is this

```python
metrics = [
    some_metric_creator(arg1, arg2)  # returns a MetricCreator, evaluated inside of the evaluator init to get a Metic
]
```

The issue here is that the user never has a chance to have a reference to a real Metric, so we can't tell them to use that to index into
things and pull resulting metrics back out of the evaluation results, if they do need to do that.

## Modules

- DONE Toxicity
- DONE Sentiment
- DONE Textstat
- DONE regexes
- DONE input_output
- DONE topics

- themes
- injections
- Hallucinations

## Existing Issues

- https://github.com/whylabs/langkit/blob/main/langkit/docs/modules.md#regexes refernces the default pattern file constantly but never
  actually links to it. What does this thing look like.
