import time
from functools import wraps
from logging import getLogger
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

logger = getLogger(__name__)

Out = TypeVar("Out")


class LazyInit(Generic[Out]):
    def __init__(self, init: Callable[[], Out]):
        self.__it: Optional[Out] = None
        self.__init = init

    @property
    def value(self) -> Out:
        if self.__it is None:
            self.__it = self.__init()
        return self.__it


In = TypeVar("In")


class DynamicLazyInit(Generic[In, Out]):
    def __init__(self, init: Callable[[In], Out]):
        self.__cache: Dict[In, Out] = {}
        self.__init = init

    def value(self, arg: In) -> Out:
        if arg not in self.__cache:
            self.__cache[arg] = self.__init(arg)

        return self.__cache[arg]


def is_dict_with_strings(variable: object) -> bool:
    if not isinstance(variable, dict):
        return False
    # Check if all values in the dictionary are strings
    return all(isinstance(value, str) for value in variable.values())  # type: ignore[reportUnknownMemberType]


ReturnType = TypeVar("ReturnType")


def retry(max_attempts: int = 3, wait_seconds: int = 1) -> Callable[[Callable[..., ReturnType]], Callable[..., ReturnType]]:
    def decorator(func: Callable[..., ReturnType]) -> Callable[..., ReturnType]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> ReturnType:
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.info(f"Attempt {attempts+1} failed with error: {e}")
                    attempts += 1
                    time.sleep(wait_seconds)
            raise Exception(f"Failed after {max_attempts} attempts")

        return wrapper

    return decorator
