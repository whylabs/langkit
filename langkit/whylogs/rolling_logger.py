import whylogs as why
from langkit import llm_metrics


class RollingLogger:
    def __init__(self):
        llm_schema = llm_metrics.init()
        self.logger = why.logger(
            mode="rolling",
            interval=5,
            when="M",
            base_name="langkit",
            schema=llm_schema,
        )
        self.logger.append_writer("whylabs")

    def log(self, dict):
        self.logger.log(dict)

    def close(self):
        self.logger.close()
