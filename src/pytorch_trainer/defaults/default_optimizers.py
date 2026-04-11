import torch.nn as nn
from torch.optim import Optimizer, AdamW, Adam
from typing import Iterable, Tuple, Dict, Type
from ..abstracts import AbstractGetter

class OptimGetter(AbstractGetter[Type[Optimizer], Optimizer]):
    
    def __init__(self, optims: Dict[str, Type[Optimizer]] | None = None) -> None:
        """Default optimizer getting object. Pass any optimizer through `optims` as a dictonary 
        of strings mapping to Type[Optimizer]. `AdamW` and `Adam` are preconfigured.

        Args:
            optims (Dict[str, Type[nn.Module]] | None, optional): Dictonary mapping optimizers to their names. Defaults to None.
        """
        super().__init__(optims)
        
        self.registry.update(
            {
                'adamw': AdamW,
                'adam': Adam
            }
        )
        
    def __call__(
        self, 
        optim_name: str,
        *, 
        model_params: Iterable[nn.Parameter] | Iterable[Tuple[str, nn.Parameter]],
        lr: float, 
        **kwargs
    ) -> Optimizer:
        kwargs['params'] = model_params
        kwargs['lr'] = lr
        return super().__call__(optim_name, **kwargs)
        