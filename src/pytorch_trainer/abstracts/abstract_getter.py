from abc import ABC
from typing import Any, Callable, Dict, Generic, TypeVar

# Makes a variable type that can be overwritten to a specific callable type
TCallable = TypeVar("TCallable", bound=Callable)
# Same thing but with return
TReturn = TypeVar("TReturn")


class AbstractGetter(ABC, Generic[TCallable, TReturn]):
    def __init__(self, to_add: Dict[str, TCallable] | None):

        self.registry: Dict[str, TCallable] = {}
        if to_add is not None:
            self.registry.update(to_add)

    def __call__(self, name: str, **kwargs: Any) -> TReturn:

        try:
            return self.get_from_registry(name, **kwargs)
        except KeyError:
            msg = (
                "{name} is not implmented for {self.__class__.__name__}."
                "Only {self.config} are valid inputs."
            )
            raise NotImplementedError(msg)

    @property
    def config(self) -> list[str]:
        return list(self.registry.keys())

    def get_from_registry(self, name: str, **kwargs: Any) -> TReturn:
        return self.registry[name](**kwargs)
