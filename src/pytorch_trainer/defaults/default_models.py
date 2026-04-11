import torch.nn as nn
from ..abstracts import AbstractModelGetter
from typing import Dict, Type

class ModelGetter(AbstractModelGetter):
    def __init__(self, models: Dict[str, Type[nn.Module]]) -> None:
        """Default model getting object. Pass any model through `models` as a dictonary 
        of strings mapping to uninstantiated models.

        Args:
            models (Dict[str, Type[nn.Module]]): Dictonary mapping models to their names.
        """
        super().__init__()
        self.model_registry: Dict[str, Type[nn.Module]] = models
        
        
    def __call__(self, model_name: str, **kwargs) -> nn.Module:
        
        try:
            return self.model_registry[model_name](**kwargs)
        except KeyError:
            raise NotImplementedError(f'{model_name} is not a configured loss function. Must be one of {list(self.model_registry.keys())}')
    
    @property
    def configured_models(self) -> list[str]:
        return list(self.model_registry.keys())