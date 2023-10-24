from typing import Any, Optional
import whylogs as why
from whylogs.core.schema import DatasetSchema
from langkit import llm_metrics


class RollingLogger:
    def __init__(
        self,
        *,
        interval_minutes: int = 5,
        schema: Optional[DatasetSchema] = None,
        **kwargs: Any
    ):
        llm_schema = llm_metrics.init()
        self.logger = why.logger(
            mode="rolling",
            interval=interval_minutes,
            when="M",
            base_name="langkit",
            schema=llm_schema,
        )
        self.logger.append_writer(name="whylabs", **kwargs)

    def log(self, dict):
        self.logger.log(dict)

    def close(self):
        self.logger.close()
