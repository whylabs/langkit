import os
from typing import Optional, Generic, Callable, Dict, TypeVar
from langkit import lang_config
import string
import random
import functools
import warnings

Out = TypeVar("Out")
In = TypeVar("In")


class DynamicLazyInit(Generic[In, Out]):
    def __init__(self, init: Callable[[In], Out]):
        self.__cache: Dict[In, Out] = {}
        self.__init = init

    def value(self, arg: In) -> Out:
        if arg not in self.__cache:
            self.__cache[arg] = self.__init(arg)

        return self.__cache[arg]


class LazyInit(Generic[Out]):
    def __init__(self, init: Callable[[], Out]):
        self.__it: Optional[Out] = None
        self.__init = init

    @property
    def value(self) -> Out:
        if self.__it is None:
            self.__it = self.__init()
        return self.__it


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
