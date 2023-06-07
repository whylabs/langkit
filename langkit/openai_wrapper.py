import openai
import contextlib
import functools
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


def log_openai_with_langkit(func, logger):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        response = func(*args, **kwargs)
        messages = kwargs.get('messages')
        if messages:
            user_prompt = [message['content'] for message in messages if message['role'] == 'user'][0]
            system_prompt = [message['content'] for message in messages if message['role'] == 'system'][0]
            llm_response = ""
            if func.__name__ == "create":
                if "ChatCompletion" in func.__qualname__:
                    llm_response = response['choices'][0]['message']['content'].strip()
                elif "Completion" in func.__qualname__:
                    llm_response = response.choices[0].text.strip()
                else:
                    llm_response = ""

            logger.log({"prompt":user_prompt, "system_prompt":system_prompt, "response":llm_response})
        return response
    return wrapper

@contextlib.contextmanager
def openai_logger():
    logger = RollingLogger()
    original_create = openai.Completion.create
    openai.Completion.create = log_openai_with_langkit(original_create, logger)
    original_chat_create = openai.ChatCompletion.create
    openai.ChatCompletion.create = log_openai_with_langkit(original_chat_create, logger)
    try:
        yield
    finally:
        openai.Completion.create = original_create
        openai.ChatCompletion.create = original_chat_create
        logger.close()
