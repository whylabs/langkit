from langkit.whylogs.rolling_logger import RollingLogger
import openai
import contextlib
import functools


def log_openai_with_langkit(func, logger):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        response = func(*args, **kwargs)
        messages = kwargs.get("messages")
        if messages:
            user_prompt = [
                message["content"] for message in messages if message["role"] == "user"
            ][0]
            system_prompt = [
                message["content"]
                for message in messages
                if message["role"] == "system"
            ][0]
            llm_response = ""
            if func.__name__ == "create":
                if "ChatCompletion" in func.__qualname__:
                    llm_response = response["choices"][0]["message"]["content"].strip()
                elif "Completion" in func.__qualname__:
                    llm_response = response.choices[0].text.strip()

            logger.log(
                {
                    "prompt": user_prompt,
                    "system_prompt": system_prompt,
                    "response": llm_response,
                }
            )
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
