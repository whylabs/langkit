import inspect
from functools import partial
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Union
from whylogs.api.logger.logger import Logger


diagnostic_logger = getLogger(__name__)


def _flex_call(func, *args, **kwargs):
    result = None
    try:
        sig = inspect.signature(func)
        params = sig.parameters
        # if params has a **kwargs style variable arguments then we don't need to
        # remove extra parameters in the filtered_kwargs below.
        has_varargs = any(param.kind == param.VAR_KEYWORD for param in params.values())

        # Helper to map position args to keyword args, so we can then check for missing arguments.
        positional_to_named_args = dict(zip(params.keys(), args))
        all_kwargs = {**positional_to_named_args, **kwargs}
        # Also remove arguments passed in that the func cannot accept
        filtered_kwargs = (
            all_kwargs
            if has_varargs
            else {k: v for k, v in all_kwargs.items() if k in params}
        )

        for key, param in params.items():
            if key not in all_kwargs and param.default is inspect.Parameter.empty:
                filtered_kwargs[key] = None
                diagnostic_logger.info(f"missing arg {key}, passing in {key}=None")

        result = func(**filtered_kwargs)
    except Exception as e:
        diagnostic_logger.warning(
            f"Error calling {func}(args{args}, kwargs{kwargs}) -> error: {e}"
        )
    return result


def _generate_callback_wrapper(handler) -> Dict[str, partial]:
    public_methods = [
        method
        for method in dir(handler)
        if callable(getattr(handler, method)) and not method.startswith("_")
    ]
    callbacks = {
        method: partial(_flex_call, getattr(handler, method))
        for method in public_methods
    }
    return callbacks


class LangKitCallback:
    def __init__(self, logger: Logger):
        """Bind the configured logger for this langKit callback handler."""
        self._logger = logger
        diagnostic_logger.info(
            f"Initialized LangKitCallback handler with configured whylogs Logger {logger}."
        )

    def _profile_generations(self, generations: List[Any]) -> None:
        for gen in generations:
            if hasattr(gen, "text"):
                self._logger.log({"response": gen.text})

    # Start LLM events
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Pass the input prompts to the logger"""
        for prompt in prompts:
            self._logger.log({"prompt": prompt})

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Pass the generated response to the logger."""
        for generations in response.generations:
            self._profile_generations(generations)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        diagnostic_logger.debug(f"on_llm_new_token({token})")

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        diagnostic_logger.debug(f"on_llm_error(error={error}, kwargs={kwargs})")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        diagnostic_logger.debug(
            f"on_chain_start(serialized={serialized}, inputs={inputs}, kwargs={kwargs})"
        )

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        diagnostic_logger.debug(f"on_chain_end(outputs={outputs}, kwargs={kwargs})")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        diagnostic_logger.debug(f"on_chain_error(error={error}, kwargs={kwargs})")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        diagnostic_logger.debug(
            f"on_chain_start(serialized={serialized}, input_str={input_str}, kwargs={kwargs})"
        )

    def on_agent_action(
        self, action: Any, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        diagnostic_logger.debug(f"on_agent_action(action={action}, kwargs={kwargs})")

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        diagnostic_logger.debug(f"on_tool_end(output={output}, kwargs={kwargs})")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        diagnostic_logger.debug(f"on_tool_error(error={error}, kwargs={kwargs})")

    def on_text(self, text: str, **kwargs: Any) -> None:
        diagnostic_logger.debug(f"on_text(text={text}, kwargs={kwargs})")

    def on_agent_finish(
        self, finish: Any, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        diagnostic_logger.debug(f"on_agent_finish(finish={finish}, kwargs={kwargs})")

    # End LLM events

    def _get_callbacks(self) -> Dict[str, partial]:
        return _generate_callback_wrapper(self)


class DynamicCallbackMeta(type):
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)

        def implement_interface(name):
            def method(self, *args, **kwargs):
                if name in self._callbacks:
                    return self._callbacks[name](*args, **kwargs)
                else:
                    return getattr(super(cls, self), name)(*args, **kwargs)

            return method

        for base in bases:
            for name, attr in base.__dict__.items():
                if callable(attr) and not name.startswith("_"):
                    setattr(cls, name, implement_interface(name))

        return cls


def DynamicCallbackAdapter(Base):
    class DynamicCallbackAdapterClass(Base, metaclass=DynamicCallbackMeta):
        # This is called by external integrations,
        # do not remove any of these parameters or add new required ones without defaults.
        def __init__(self, whylabs_logger: Logger, handler: Any):
            if hasattr(handler, "init"):
                handler.init(self)
            if hasattr(handler, "_get_callbacks"):
                self._callbacks = handler._get_callbacks()
                diagnostic_logger.debug(
                    f"initialized LangKit handler with {self._callbacks}."
                )
            else:
                self._callbacks = dict()
                diagnostic_logger.warning(
                    "initialized LangKit handler without callbacks."
                )
            self._methods: Dict[str, Callable] = dict()
            self._logger = whylabs_logger

        def __getattr__(self, name):
            if name in self._callbacks:
                return self._callbacks[name]

            if name in self._methods:
                return self._methods[name]

            def no_op_method(*args, **kwargs):
                diagnostic_logger.debug(
                    f"no passthrough for '{name}' this event, args={args},kwargs={kwargs}."
                )

            self._methods[name] = no_op_method
            return no_op_method

    return DynamicCallbackAdapterClass


def get_callback_instance(*args, **kwargs):
    handler = kwargs.get("handler")
    logger = kwargs.get("logger")
    if handler is None:
        logger = kwargs.get("logger")
        handler = LangKitCallback(logger=logger)
    elif logger is None:
        logger = handler._logger
    base_class = handler.__class__
    impl = kwargs.get("impl")
    LangKitCallbackImplementation = DynamicCallbackAdapter(base_class)
    if impl:
        LangKitCallbackImplementation.__bases__ += (impl,)
    return LangKitCallbackImplementation(logger, handler=handler)
