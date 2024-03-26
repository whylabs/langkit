from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd

from langkit.core.metric import Metric, MultiMetric, SingleMetric, WorkflowMetricConfig
from whylogs.core.resolvers import StandardMetric
from whylogs.core.segmentation_partition import SegmentationPartition
from whylogs.experimental.core.metrics.udf_metric import MetricConfig as YMetricConfig
from whylogs.experimental.core.metrics.udf_metric import TypeMapper
from whylogs.experimental.core.udf_schema import NO_FI_RESOLVER, MetricSpec, ResolverSpec, UdfSchema, UdfSpec, Validator


@dataclass(frozen=True)
class UdfSchemaArgs:
    """
    This shouldn't really exist. It does because creating a UdfSchema ends up
    losing references to these things, which means you can't get them back out when you're
    trying to combine schemas later. Instead of passing this to the UdfSchema constructor,
    we save it in this thing.
    """

    resolvers: Optional[List[ResolverSpec]] = None
    types: Optional[Dict[str, Any]] = None
    default_config: Optional[YMetricConfig] = None
    type_mapper: Optional[TypeMapper] = None
    cache_size: int = 1024
    schema_based_automerge: bool = False
    segments: Optional[Dict[str, SegmentationPartition]] = None
    validators: Optional[Dict[str, List[Validator]]] = None
    udf_specs: Optional[List[UdfSpec]] = None


def _to_udf_schema_args_single(metric: SingleMetric) -> UdfSchemaArgs:
    # TODO evaluate these names and make sure they match up with the metric names that we end up with, and unit test it.
    # This entire function just hard coded looks for metric names and knows how to translate things.
    def udf(text: Union[pd.DataFrame, Dict[str, List[Any]]]) -> Any:
        if isinstance(text, pd.DataFrame):
            return metric.evaluate(text).metrics
        else:
            return metric.evaluate(pd.DataFrame(text)).metrics

    if "has_patterns" in metric.name or "closest_topic" in metric.name:
        resolvers = [
            ResolverSpec(
                column_name=metric.name,
                metrics=[MetricSpec(StandardMetric.frequent_items.value)],
            )
        ]
    else:
        # We only want to add resolvers here if they're different from the default ones. Otherwise, we'll
        # load up the final schema with a bunch of standard resolvers to cover all normal cases and avoid
        # duplicates
        resolvers = []

    if "similarity.prompt" in metric.name:
        # This is the only way to make it workout correctlyfor input_output_similarity, which is fine for now. Generally, we'll
        # need metrics to be able to declare multiple inputs if this is going to work for custom metrics. We can probably continue
        # to assume str type for inputs though.
        types = {"prompt": str, "response": str}
        column_names = ["prompt", "response"]
    else:
        types: Dict[str, Type[str]] = {}
        for name in metric.input_names:
            types[name] = str

        column_names = [*metric.input_names]

    schema = UdfSchemaArgs(
        types=types,
        resolvers=resolvers,
        udf_specs=[
            UdfSpec(
                column_names=column_names,
                udfs={metric.name: udf},
            )
        ],
    )

    return schema


def _to_udf_schema_args_multiple(metric: MultiMetric) -> UdfSchemaArgs:
    def udf(text: Union[pd.DataFrame, Dict[str, List[Any]]]) -> Any:
        wf_input = pd.DataFrame(text) if isinstance(text, dict) else text
        metrics = metric.evaluate(wf_input).metrics
        return pd.concat([pd.Series(metric, name=name) for (metric, name) in zip(metrics, metric.names)], axis=1)  # pyright: ignore [reportUnknownMemberType]

    return UdfSchemaArgs(
        resolvers=[],
        types={k: str for k in metric.names},
        udf_specs=[UdfSpec(column_names=metric.input_names, udf=udf, prefix="")],
    )


def to_udf_schema_args(metric: Metric) -> List[UdfSchemaArgs]:
    if isinstance(metric, SingleMetric):
        return [_to_udf_schema_args_single(metric)]
    else:
        return [_to_udf_schema_args_multiple(metric)]


def create_whylogs_udf_schema(eval_conf: WorkflowMetricConfig) -> UdfSchema:
    for metric in eval_conf.metrics:
        if metric.init:
            metric.init()

    metrics = [to_udf_schema_args(it) for it in eval_conf.metrics]
    flattened_metrics = reduce(lambda a, b: a + b, metrics)
    args = reduce(combine_schemas, flattened_metrics, UdfSchemaArgs(resolvers=NO_FI_RESOLVER))

    return UdfSchema(
        resolvers=args.resolvers,
        types=args.types,
        default_config=args.default_config,
        type_mapper=args.type_mapper,
        cache_size=args.cache_size,
        schema_based_automerge=args.schema_based_automerge,
        segments=args.segments,
        validators=args.validators,
        udf_specs=args.udf_specs,
    )


def combine_type_mappers(a: TypeMapper, b: TypeMapper) -> TypeMapper:
    # TODO implement
    return a


def combine_schemas(a: UdfSchemaArgs, b: UdfSchemaArgs) -> UdfSchemaArgs:
    return UdfSchemaArgs(
        resolvers=[*(a.resolvers or []), *(b.resolvers or [])],
        types={**(a.types or {}), **(b.types or {})},
        default_config=a.default_config if a.default_config is not None else b.default_config,
        type_mapper=combine_type_mappers(a.type_mapper, b.type_mapper) if a.type_mapper is not None and b.type_mapper is not None else None,
        cache_size=max(a.cache_size, b.cache_size),  # TODO verify this is correct
        schema_based_automerge=a.schema_based_automerge or b.schema_based_automerge,  # TODO verify this is correct
        segments=a.segments if a.segments is not None else b.segments,
        validators={**(a.validators or {}), **(b.validators or {})},
        udf_specs=[*(a.udf_specs or []), *(b.udf_specs or [])],
    )
