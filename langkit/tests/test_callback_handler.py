from logging import getLogger
from typing import Any, Dict, List
from langkit.callback_handler import LangKitCallback, get_callback_instance


TEST_LOGGER = getLogger(__name__)


class MockLogger:
    def __getattr__(self, name):
        def method(*args, **kwargs):
            TEST_LOGGER.info(f"logger called {name}(*args={args}, **kwargs={kwargs})")

        return method


class MockCallbackOnStartMixin1:
    def on_llm_start(self, prompts: List[str]):
        TEST_LOGGER.info(
            f"MockCallbackOnStartMixin1.on_llm_start called on_llm_start with {prompts}"
        )


class MockCallbackOnStartMixin2:
    def on_llm_end(self, response):
        TEST_LOGGER.info(
            f"MockCallbackOnStartMixin2.on_llm_end called on_llm_start with response={response}"
        )


class MockCallbackOnStartMixin3:
    def on_text(self, prompts: List[str]):
        TEST_LOGGER.info(
            f"MockCallbackOnStartMixin3.on_text called on_llm_start with {prompts}"
        )


class ComplexBaseHandler(
    MockCallbackOnStartMixin1, MockCallbackOnStartMixin2, MockCallbackOnStartMixin3
):
    def ignore_llm(self):
        TEST_LOGGER.info("Calling ignore_llm LangChainBaseHandler")


class MockBaseHandler:
    def close(self):
        TEST_LOGGER.info("Calling close in test MockBaseHandler")


class MockBaseHandler2:
    def close(self):
        TEST_LOGGER.info("Calling close in test MockBaseHandler2")


def test_callback_passthroughs_undefined_ok():
    universal_callback = get_callback_instance()
    universal_callback.foo(a="hi", b=True)
    foo1 = universal_callback.foo
    foo2 = universal_callback.foo
    assert foo1 is foo2


def test_callback_passthroughs_undefined_no_args():
    universal_callback = get_callback_instance()
    universal_callback.bar()
    universal_callback.baz()


def test_callback_passthroughs_defined_functions():
    universal_callback = get_callback_instance()
    universal_callback.on_text(text="Hello texty text!")


def test_callback_passthroughs_defined_logging_functions():
    universal_callback = get_callback_instance(
        logger=MockLogger(), impl=MockBaseHandler, interface=MockCallbackOnStartMixin1
    )
    test_prompts = ["hi"]
    default_serialized: Dict[str, Any] = {"test": "serialized"}
    on_llm_start = universal_callback.on_llm_start
    universal_callback.on_llm_start(serialized=default_serialized, prompts=test_prompts)
    on_llm_start(default_serialized, prompts=test_prompts)
    test_response = type("", (object,), {"generations": [{"text": "No"}]})()
    universal_callback.on_llm_end(response=test_response)
    universal_callback.close()


def test_callback_instance_handler_defined():
    callback_handler = LangKitCallback(logger=MockLogger())
    universal_callback = get_callback_instance(
        handler=callback_handler, impl=MockBaseHandler2
    )
    test_prompts = ["goodbye!"]
    universal_callback.on_llm_start(prompts=test_prompts)
    universal_callback.close()


def test_callback_instance_handler_with_metadata():
    callback_handler = LangKitCallback(logger=MockLogger())
    universal_callback = get_callback_instance(
        handler=callback_handler, impl=MockBaseHandler2
    )
    universal_callback.include_metadata()
    test_prompts = ["goodbye!"]
    universal_callback.on_llm_start(prompts=test_prompts)
    mock_response = type("MockResponse", (object,), {"generations": []})
    universal_callback.on_llm_end(response=mock_response)


def test_callback_instance_handler_defined_getattr():
    callback_handler = LangKitCallback(logger=MockLogger())
    universal_callback = get_callback_instance(
        handler=callback_handler, impl=MockBaseHandler2, base=ComplexBaseHandler
    )
    test_prompts = ["goodbye variations!"]
    method_name = "on_llm_start"

    assert hasattr(universal_callback, method_name)
    getattr_method = getattr(universal_callback, method_name)
    direct_method = universal_callback.on_llm_start
    TEST_LOGGER.info(
        f"comparing getattr with method name {getattr_method} vs {direct_method}"
    )
    getattr_method(prompts=test_prompts)
    direct_method(prompts=test_prompts)
    universal_callback.close()


def test_callback_instance_three_ply_class_hierarchy():
    callback_handler = LangKitCallback(logger=MockLogger())
    universal_callback = get_callback_instance(
        handler=callback_handler, impl=MockBaseHandler2, base=ComplexBaseHandler
    )
    test_prompts = ["goodbye variations!"]
    method_name = "on_llm_start"

    assert hasattr(universal_callback, method_name)
    getattr_method = getattr(universal_callback, method_name)
    direct_method = universal_callback.on_llm_start
    TEST_LOGGER.info(
        f"comparing getattr with method name {getattr_method} vs {direct_method}"
    )
    getattr_method(prompts=test_prompts)
    direct_method(prompts=test_prompts)
    universal_callback.close()
