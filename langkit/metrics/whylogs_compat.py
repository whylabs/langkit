from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from langkit.metrics.metric import EvaluationConfig, Metric, SingleMetric, SingleMetricResult
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
        resolvers = NO_FI_RESOLVER

    if "relevance_to_prompt" in metric.name:
        # This is the only way to make it workout correctlyfor input_output_similarity, which is fine for now
        types = {"prompt": str, "response": str}
        column_names = ["prompt", "response"]
    else:
        types = {metric.input_name: str}
        column_names = [metric.input_name]

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


def to_udf_schema_args(metric: Metric) -> List[UdfSchemaArgs]:
    metrics: List[SingleMetric] = []

    if isinstance(metric, SingleMetric):
        metrics.append(metric)
    else:
        for i, name in enumerate(metric.names):
            # Whylogs doesn't support multi-metrics, so we have to convert them to single metrics. This is lame because
            # the only real way to do this is to re-evaluate the metric for each name, which is wasteful, but at least
            # its possible. It just won't be advised to use multi metrics when using whylogs.
            def _lame_udf(text: pd.DataFrame) -> SingleMetricResult:
                result = metric.evaluate(text)
                return SingleMetricResult(metrics=result.metrics[i])

            metrics.append(SingleMetric(name=name, input_name=metric.input_name, evaluate=_lame_udf))

    return [_to_udf_schema_args_single(it) for it in metrics]


def create_whylogs_udf_schema(eval_conf: EvaluationConfig) -> UdfSchema:
    metrics = [to_udf_schema_args(it) for it in eval_conf.metrics]
    flattened_metrics = reduce(lambda a, b: a + b, metrics)
    args = reduce(combine_schemas, flattened_metrics)

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
        resolvers=a.resolvers + b.resolvers if a.resolvers is not None and b.resolvers is not None else None,
        types=a.types if a.types is not None else b.types,
        default_config=a.default_config if a.default_config is not None else b.default_config,
        type_mapper=combine_type_mappers(a.type_mapper, b.type_mapper) if a.type_mapper is not None and b.type_mapper is not None else None,
        cache_size=max(a.cache_size, b.cache_size),  # TODO verify this is correct
        schema_based_automerge=a.schema_based_automerge or b.schema_based_automerge,  # TODO verify this is correct
        segments=a.segments if a.segments is not None else b.segments,
        validators=a.validators if a.validators is not None else b.validators,
        udf_specs=a.udf_specs + b.udf_specs if a.udf_specs is not None and b.udf_specs is not None else None,
    )