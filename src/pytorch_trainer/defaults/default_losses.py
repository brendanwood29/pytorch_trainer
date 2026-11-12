from typing import Dict, Type

import torch.nn as nn

from ..abstracts import AbstractGetter


class LossGetter(AbstractGetter[Type[nn.Module], nn.Module]):
    def __init__(self, loss_fns: Dict[str, Type[nn.Module]] | None = None):
        """Default loss getting object. Pass any loss functions through `loss_fns`
        as a dictionary of strings mapping to Type[nn.Module].
        `MSELoss` and `BCELoss` are preconfigured.

        Args:
            loss_fns (Dict[str, Type[nn.Module]] | None, optional):
                Dictonary mapping loss functions to their names. Defaults to None.
        """
        super().__init__(loss_fns)
        self.registry.update({"mse_loss": nn.MSELoss, "bce_loss": nn.BCELoss})

    def __call__(self, loss_name: str, **kwargs) -> nn.Module:
        return super().__call__(loss_name, **kwargs)
