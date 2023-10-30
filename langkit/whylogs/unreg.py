import whylogs.experimental.core.udf_schema as us
from typing import Optional, Set
import logging

logger = logging.getLogger(__name__)


def unregister_udf(
    udf_name: str, namespace: Optional[str] = None, schema_name: str = ""
) -> None:
    name = f"{namespace}.{udf_name}" if namespace else udf_name
    if schema_name not in us._multicolumn_udfs:
        logger.warn(
            f"Can't unregister UDF {name} from non-existant schema {schema_name}"
        )
        return

    found = False
    for spec in us._multicolumn_udfs[schema_name]:
        if name in spec.udfs:
            found = True
            del spec.udfs[name]
    if not found:
        logger.warn(f"UDF {name} could not be found for unregistering")
    us._resolver_specs[schema_name] = list(
        filter(lambda x: x.column_name != name, us._resolver_specs[schema_name])
    )


def unregister_udfs(
    udfs: Set[str], namespace: Optional[str] = None, schema_name: str = ""
) -> None:
    for udf in udfs:
        unregister_udf(udf, namespace, schema_name)
    udfs = set()
