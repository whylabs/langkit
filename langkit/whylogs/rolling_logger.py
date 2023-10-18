from typing import Any
import whylogs as why
from langkit import llm_metrics


class RollingLogger:
    def __init__(self, **kwargs: Any):
        llm_schema = llm_metrics.init()
        self.logger = why.logger(
            mode="rolling",
            interval=5,
            when="M",
            base_name="langkit",
            schema=llm_schema,
        )
        self.logger.append_writer(name="whylabs", **kwargs)

    def log(self, dict):
        self.logger.log(dict)

    def close(self):
        self.logger.close()
