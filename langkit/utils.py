import os
from typing import Optional
from langkit import lang_config
import string
import random
import functools
import warnings


def deprecated(message):
    def decorator_deprecated(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator_deprecated


def _get_data_home() -> str:
    package_dir = os.path.dirname(__file__)
    data_path = os.path.join(package_dir, lang_config.data_folder)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    return data_path


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def _unregister_metric_udf(old_name: str, namespace: Optional[str] = ""):
    from whylogs.experimental.core.udf_schema import _multicolumn_udfs

    if _multicolumn_udfs is None or namespace not in _multicolumn_udfs:
        return

    new_multicolumn_udfs = []
    for udf in _multicolumn_udfs[namespace]:
        udfs_cols = list(udf.udfs.keys())
        if udfs_cols and udfs_cols[0] != old_name:
            new_multicolumn_udfs.append(udf)
        # added to account for multicolumn udfs
        elif udf.udf and udf.prefix != old_name:
            new_multicolumn_udfs.append(udf)

    _multicolumn_udfs[namespace] = new_multicolumn_udfs
