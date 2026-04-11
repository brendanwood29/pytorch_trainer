from typing import Dict, Callable, TypeVar, Generic, Any
from abc import ABC

TCallable = TypeVar('TCallable', bound=Callable)
TReturn = TypeVar('TReturn')

class AbstractGetter(ABC, Generic[TCallable, TReturn]):
    def __init__(self, to_add: Dict[str, TCallable] | None):
        
        self.registry: Dict[str, TCallable] = {}
        if to_add is not None:
            self.registry.update(to_add)
        
    def __call__(self, name: str, **kwargs: Any) -> TReturn:
        
        try:
            return self.get_from_registry(name, **kwargs)
        except KeyError:
            raise NotImplementedError(f'{name} is not implmented for {self.__class__.__name__}. Only {self.config} are valid inputs.')
        
    @property
    def config(self) -> list[str]:
        return list(self.registry.keys())
    
    def get_from_registry(self, name: str, **kwargs: Any) -> TReturn:
        return self.registry[name](**kwargs)