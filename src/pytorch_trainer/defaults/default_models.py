import torch.nn as nn
from ..abstracts import AbstractGetter
from typing import Dict, Type

class ModelGetter(AbstractGetter[Type[nn.Module], nn.Module]):
    def __init__(self, models: Dict[str, Type[nn.Module]]) -> None:
        """Default model getting object. Pass any model through `models` as a dictonary 
        of strings mapping to Type[nn.Module].

        Args:
            models (Dict[str, Type[nn.Module]]): Dictonary mapping models to their names.
        """
        super().__init__(models)
        
    def __call__(self, model_name: str, **kwargs) -> nn.Module:
        return super().__call__(model_name, **kwargs)