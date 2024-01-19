from typing import Callable, Dict, Generic, Optional, TypeVar

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
