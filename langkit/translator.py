from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
from whylogs.core.stubs import pd as pd


class Translator(ABC):
    @abstractmethod
    def translate(self, text: str) -> str:
        return text


def translated(translator: Optional[Translator] = None):
    def decorator(func: Callable[[str], Any]) -> Callable[[str], Any]:
        return lambda text: (func(translator.translate(text)) if translator else text)

    return decorator


def translated_udf(translators: Optional[Dict[str, Optional[Translator]]]):
    def decorator(
        func: Callable[[Union[Dict[str, List], pd.DataFrame]], Union[List, pd.Series]]
    ) -> Callable[[Union[Dict[str, List], pd.DataFrame]], Union[List, pd.Series]]:
        def wrapper(
            text: Union[Dict[str, List], pd.DataFrame]
        ) -> Union[List, pd.Series]:
            if translators is None:
                return text

            if isinstance(text, dict):
                translated = {
                    k: [
                        (translators[k].translate(t) if translators[k] else t)  # type: ignore
                        for t in v
                    ]
                    for k, v in text.items()
                }
                return func(translated)

            translated = pd.DataFrame()
            for k in text.keys():
                translated[k] = (
                    text[k].map(translators[k].translate) if translators[k] else text[k]  # type: ignore
                )
            return func(translated)

        return wrapper

    return decorator
