from __future__ import annotations

from dataclasses import dataclass
from functools import partial, reduce
from typing import Any, Callable, Dict, Iterator, List, Optional, Union, cast

import pandas as pd

from whylogs.core.resolvers import Resolver
from whylogs.core.schema import DatasetSchema
from whylogs.core.segmentation_partition import SegmentationPartition
from whylogs.experimental.core.metrics.udf_metric import MetricConfig, TypeMapper
from whylogs.experimental.core.udf_schema import NO_FI_RESOLVER, ResolverSpec, UdfSchema, UdfSpec, Validator


# TODO make this generic and add a filter ability to ensure that it only delivers the things
# you want instead of a bunch of Any
class UdfInput:
    def __init__(self, text: Union[pd.DataFrame, Dict[str, List[Any]]]) -> None:
        self.text = text

    def iter_column(self, column_name: str) -> Iterator[Any]:
        if column_name not in self.text:
            return iter([])

        if isinstance(self.text, pd.DataFrame):
            col = cast(pd.Series, self.text[column_name])
            return cast(Iterator[Any], iter(col))
        else:
            return iter(self.text[column_name])


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
    default_config: Optional[MetricConfig] = None
    type_mapper: Optional[TypeMapper] = None
    cache_size: int = 1024
    schema_based_automerge: bool = False
    segments: Optional[Dict[str, SegmentationPartition]] = None
    validators: Optional[Dict[str, List[Validator]]] = None
    udf_specs: Optional[List[UdfSpec]] = None


UdfFunctionInput = Union[pd.DataFrame, Dict[str, List[Any]]]

ColumnName = str
UdfFunction = Callable[[ColumnName, UdfFunctionInput], Union[pd.DataFrame, Dict[str, List[Any]]]]


class ModuleBuilder:
    def __init__(self) -> None:
        self.args: List[UdfSchemaArgs] = []

    def add_udf(self, input_column_name: str, output_column_name: str, udf: UdfFunction) -> "ModuleBuilder":
        _udf = partial(udf, input_column_name)

        udf_spec = UdfSpec(
            column_names=[input_column_name],
            udfs={output_column_name: _udf},
        )

        schema = UdfSchemaArgs(
            types={input_column_name: str},
            resolvers=NO_FI_RESOLVER,  # TODO is this the right default resolver to use?
            udf_specs=[udf_spec],
        )

        self.args.append(schema)

        return self

    # TODO add a fn for adding conditions/validators
    # def build(self) -> "RegexValidatorBuilder":
    #     from langkit import regexes  # pyright: ignore reportUnusedImport
    #
    #     config_path: Optional[str] = self.validation_rule.config_path
    #     if config_path is not None:
    #         regexes.init(pattern_file_path=config_path)  # type: ignore
    #     self.key = f"{self.validation_rule.rule_type}.has_patterns"
    #
    #     def validate_patterns_condition(pattern: str) -> bool:
    #         return not bool(pattern)
    #
    #     passed_conditions: Dict[str, Union[Condition, Callable[[Any], bool]]] = {
    #         "no_patterns": Condition(Predicate().is_(validate_patterns_condition))
    #     }
    #     self.validator: ConditionValidator = ConditionValidator(
    #         name="no_patterns_validator",
    #         conditions=passed_conditions,
    #         actions=[flag_failed_validation],
    #     )
    #     return self
    #

    def build(self) -> "Module":
        return lambda: self.args


ModuleFn = Callable[[], UdfSchemaArgs]
Module = Union[ModuleFn, Callable[[], List[UdfSchemaArgs]], List[ModuleFn]]


class SchemaBuilder:
    def __init__(self) -> None:
        super().__init__()
        self._modules: List[Module] = []

    def add(self, module: Module) -> "SchemaBuilder":
        if isinstance(module, list):
            self._modules.extend(module)
        elif callable(module):
            self._modules.append(module)
        else:
            self._modules.extend(module)

        return self

    def build(self) -> DatasetSchema:
        schemas: List[UdfSchemaArgs] = []
        for module in self._modules:
            if callable(module):
                schema = module()
                if isinstance(schema, UdfSchemaArgs):
                    schemas.append(schema)
                else:
                    schemas.extend(schema)
            else:
                # schemas.extend([m.create() for m in module])
                for m in module:
                    if isinstance(m, ModuleBuilder):
                        schemas.append(m.build())
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
