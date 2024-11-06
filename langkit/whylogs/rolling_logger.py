from typing import Any, Optional
import whylogs as why
from whylogs.core.schema import DatasetSchema


class RollingLogger:
    def __init__(
        self,
        *,
        interval_minutes: int = 5,
        schema: Optional[DatasetSchema] = None,
        **kwargs: Any
    ):
        if schema is None:
            from langkit import llm_metrics

            schema = llm_metrics.init()
        self.logger = why.logger(
            mode="rolling",
            interval=interval_minutes,
            when="M",
            base_name="langkit",
            schema=schema,
        )

        if "writer" in kwargs:
            self.logger.append_writer(name=None, **kwargs)
        else:
            self.logger.append_writer(name="whylabs", **kwargs)

    def log(self, dict):
        self.logger.log(dict)

    def close(self):
        self.logger.close()
