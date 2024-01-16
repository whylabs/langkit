import os
from typing import Optional
from langkit import lang_config
import string
import random


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

    _multicolumn_udfs[namespace] = [
        udf
        for udf in _multicolumn_udfs[namespace]
        if list(udf.udfs.keys())[0] != old_name
    ]
