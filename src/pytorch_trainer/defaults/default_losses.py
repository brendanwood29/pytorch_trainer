import torch.nn as nn
from ..abstracts import AbstractLossGetter
from typing import Dict, Type

class LossGetter(AbstractLossGetter):
    
    def __init__(self, loss_funcs: Dict[str, Type[nn.Module]] | None = None) -> None:
        """Default loss getting object. Pass any loss functions through `loss_funcs` as a dictonary 
        of strings mapping to uninstantiated loss functions. `MSELoss` and `BCELoss` are preconfigured.

        Args:
            loss_funcs (Dict[str, Type[nn.Module]] | None, optional): Dictonary mapping loss functions to their names. Defaults to None.
        """
        super().__init__()
        
        self.loss_func_registry: Dict[str, Type[nn.Module]] = {
            'mse_loss': nn.MSELoss,
            'bce_loss': nn.BCELoss
        }
        if loss_funcs is not None:
            self.loss_func_registry.update(loss_funcs)
        
    def __call__(self, loss_name: str, **kwargs) -> nn.Module:
        
        try:
            return self.loss_func_registry[loss_name](**kwargs)
        except KeyError:
            raise NotImplementedError(f'{loss_name} is not a configured loss function. Must be one of {list(self.loss_func_registry.keys())}')
    
    @property
    def configured_losses(self) -> list[str]:
        return list(self.loss_func_registry.keys())
