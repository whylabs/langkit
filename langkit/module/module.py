from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Union

from whylogs.core.resolvers import Resolver
from whylogs.core.schema import DatasetSchema
from whylogs.core.segmentation_partition import SegmentationPartition
from whylogs.experimental.core.metrics.udf_metric import MetricConfig, TypeMapper
from whylogs.experimental.core.udf_schema import ResolverSpec, UdfSchema, UdfSpec, Validator


@dataclass
class UdfSchemaArgs:
    """
    This shouldn't really exist. It does because creating a UdfSchema ends up
    losing references to these things, which means you can't get them back out when you're
    trying to combine schemas later. Instead of passing this to the UdfSchema constructor,
    we save it in this thing.
    """

    resolvers: Optional[List[ResolverSpec]] = None
    types: Optional[Dict[str, Any]] = None
    default_config: Optional[MetricConfig] = None
    type_mapper: Optional[TypeMapper] = None
    cache_size: int = 1024
    schema_based_automerge: bool = False
    segments: Optional[Dict[str, SegmentationPartition]] = None
    validators: Optional[Dict[str, List[Validator]]] = None
    udf_specs: Optional[List[UdfSpec]] = None


class Module(ABC):
    @abstractmethod
    def create(self) -> UdfSchemaArgs:
        pass


ModuleType = Union[Module, List[Module], Callable[[], UdfSchemaArgs], Callable[[], List[UdfSchemaArgs]], List[Callable[[], UdfSchemaArgs]]]


class SchemaBuilder:
    def __init__(self) -> None:
        super().__init__()
        self._modules: List[ModuleType] = []

    def add(self, module: ModuleType) -> "SchemaBuilder":
        if isinstance(module, Module):
            self._modules.append(module)
        elif isinstance(module, list):
            self._modules.extend(module)
        elif callable(module):
            self._modules.append(module)
        else:
            self._modules.extend(module)

        return self

    def build(self) -> DatasetSchema:
        schemas: List[UdfSchemaArgs] = []
        for module in self._modules:
            if isinstance(module, Module):
                schemas.append(module.create())
            elif callable(module):
                schema = module()
                if isinstance(schema, UdfSchemaArgs):
                    schemas.append(schema)
                else:
                    schemas.extend(schema)
            else:
                # schemas.extend([m.create() for m in module])
                for m in module:
                    if isinstance(m, Module):
                        schemas.append(m.create())
                    elif callable(m):
                        schema = m()
                        schemas.append(schema)

        # reduce the schemas
        args = reduce(combine_schemas, schemas)
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


def combine_metric_configs(a: MetricConfig, b: MetricConfig) -> MetricConfig:
    # TODO implement
    return a


def combine_resolvers(a: Resolver, b: Resolver) -> Resolver:
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
