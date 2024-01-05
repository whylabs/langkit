from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")


class LazyInit(Generic[T]):
    def __init__(self, init: Callable[[], T]):
        self.__it: Optional[T] = None
        self.__init = init

    @property
    def value(self) -> T:
        if self.__it is None:
            self.__it = self.__init()
        return self.__it
