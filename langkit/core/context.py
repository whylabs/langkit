from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, TypeVar

import pandas as pd


@dataclass
class Context:
    request_data: Dict[str, Any] = field(default_factory=dict)


RequestData = TypeVar("RequestData")


class ContextDependency(Generic[RequestData]):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def cache_assets(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def init(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def populate_request(self, context: Context, data: pd.DataFrame) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_request_data(self, context: Context) -> RequestData:
        raise NotImplementedError()
