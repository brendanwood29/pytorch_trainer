from torch import nn
from abc import ABC, abstractmethod

class LossGetter(ABC):
    
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def __call__(self, loss_name: str, **kwargs) -> nn.Module:
        """Overwrite to return the loss function. If you define your own, make it a subclass of nn.Module.

        Args:
            loss_name (str): Loss name.
            kwargs: Keyword arguments to be passed to the loss function.

        Raises:
            NotImplementedError: If you do not add your new loss function to the list of configured loss functions.

        Returns:
            nn.Module: Callable loss function.
        """
        
        configured_losses = [
            'mse_loss',
            'mse_fc_loss'
        ]
        
        if loss_name not in configured_losses:
            raise NotImplementedError(f'{loss_name} is not a configured loss function, `name` must be one of {configured_losses}')
        
        if loss_name == 'mse_loss':
            return nn.MSELoss()
        
        
    