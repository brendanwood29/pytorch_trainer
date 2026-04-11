import torch.nn as nn
from torch.optim import Optimizer, AdamW, Adam
from typing import Iterable, Tuple, Dict, Type
from ..abstracts import AbstractOptimGetter

class OptimGetter(AbstractOptimGetter):
    
    def __init__(self, optims: Dict[str, Type[Optimizer]] | None = None) -> None:
        """Default optimizer getting object. Pass any optimizer through `optims` as a dictonary 
        of strings mapping to uninstantiated optimizers. `AdamW` and `Adam` are preconfigured.

        Args:
            optims (Dict[str, Type[nn.Module]] | None, optional): Dictonary mapping optimizers to their names. Defaults to None.
        """
        super().__init__()
        
        self.optim_registry: Dict[str, Type[Optimizer]] = {
            'adamw': AdamW,
            'adam': Adam
        }
        if optims is not None:
            self.optim_registry.update(optims)
        
    def __call__(
        self, 
        optim_name: str, 
        model_params: Iterable[nn.Parameter] | Iterable[Tuple[str, nn.Parameter]],
        lr: float, 
        **kwargs) -> Optimizer:
        
        try:
            return self.optim_registry[optim_name](model_params, lr=lr, **kwargs) # type: ignore[call-arg]  # registry stores Optimizer subclasses which all should accept lr
        except KeyError:
            raise NotImplementedError(f'{optim_name} is not a configured optimizer. Must be one of {list(self.optim_registry.keys())}')
        
    @property
    def configured_optims(self) -> list[str]:
        return list(self.optim_registry.keys())