import whylogs as why
from whylogs.experimental.core.udf_schema import udf_schema

class RollingLogger:
    def __init__(self):
        self.logger = why.logger(mode="rolling", interval=5, when="M", base_name="langkit", schema=udf_schema())
        self.logger.append_writer("whylabs")
    
    def log(self, dict):
        self.logger.log(dict)
    
    def close(self):
        self.logger.close()
