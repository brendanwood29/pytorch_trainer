import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class MLP(nn.Module):
    
    def __init__(
        self,
        in_feats: int=5,
        out_feats: int=5,
    ) -> None:
        super().__init__()
        
        self.lin = nn.Linear(in_feats, out_feats)
        self.activation = nn.ReLU()
        
    def forward(self, x) -> torch.Tensor:
        
        return self.activation(self.lin(x))


class ModelGetter(ABC):
    
    def __init__(self) -> None:
        
        super().__init__()
    
    
    @abstractmethod
    def __call__(self, model_name: str, **kwargs) -> nn.Module:
        """Overwrite this method to return your model.

        Args:
            model_name (str): Name of model.
            kwargs: Any keword arguments to be passed to the model.

        Raises:
            NotImplementedError: If you do not add your new model to the list of configured models.

        Returns:
            nn.Module: Your configued model.
        """
        configured_models = [
            'mlp',
        ]
        
        if model_name not in configured_models:
            raise NotImplementedError(f'{model_name} is not a configured model, `name` must be one of {configured_models}')
        
        if model_name == 'mlp':
            return MLP(
                **kwargs
            )
