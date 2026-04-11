import torch.nn as nn
from torch.optim import Optimizer
from typing import Iterable, Tuple
from abc import ABC, abstractmethod


class AbstractOptimGetter(ABC):
    
    def __init__(self) -> None:
        super().__init__()
    
    
    @abstractmethod
    def __call__(
        self, 
        optim_name: str, 
        model_params: Iterable[nn.Parameter] | Iterable[Tuple[str, nn.Parameter]], 
        lr: float, 
        **kwargs
        ) -> Optimizer:
        """Overwrite this method to return your configured optimizer.

        Args:
            optim_name (str): Name of optimizer.
            model_params (Iterable[nn.Parameter] | Iterable[Tuple[str, nn.Parameter]]): Model parameters. Typically `model.parameters()` or `model.named_parameters()`
            lr (float): Learning rate.

        Raises:
            NotImplementedError: If optimizer name is not in your configured optimizers.

        Returns:
            Optimizer: Configured optimizer.
        ```python
            configured_optimizers = [
                'adamw',
            ]
            
            if optim_name not in configured_optimizers:
                raise NotImplementedError(f'{optim_name} is not a configured scheduler, `optim_name` must be one of {configured_optimizers}')
            
            if optim_name == 'adamw':
                
                return AdamW(
                    params=model_params,
                    lr=lr,
                    **kwargs
                )
        ```
        """
            