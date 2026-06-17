from typing import Dict, Type

from torch.optim.lr_scheduler import LRScheduler, OneCycleLR, StepLR
from torch.optim.optimizer import Optimizer

from ..abstracts import AbstractGetter


class SchedulerGetter(AbstractGetter[Type[LRScheduler], LRScheduler]):
    def __init__(self, schedulers: Dict[str, Type[LRScheduler]] | None = None) -> None:
        """Default scheduler getting object. Pass any scheduler functions through
        `schedulers` as a dictionary of strings mapping to Type[LRScheduler].
        `OneCycleLR` and `StepLR` are configured.

        Args:
            schedulers (Dict[str, Type[LRScheduler]] | None, optional):
                Dictonary mapping schedulers to their names. Defaults to None.
        """
        super().__init__(schedulers)

        self.registry.update(
            {
                "onecycle": OneCycleLR,
                "step": StepLR,
            }
        )

    def __call__(
        self, scheduler_name: str, *, optim: Optimizer, **kwargs
    ) -> LRScheduler:

        kwargs["optimizer"] = optim
        return super().__call__(scheduler_name, **kwargs)
