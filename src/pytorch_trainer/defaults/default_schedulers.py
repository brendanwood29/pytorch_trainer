from torch.optim.lr_scheduler import LRScheduler, OneCycleLR, StepLR
from torch.optim.optimizer import Optimizer
from ..abstracts import AbstractSchedulerGetter
from typing import Dict, Type

class SchedulerGetter(AbstractSchedulerGetter):
    
    def __init__(self, schedulers: Dict[str, Type[LRScheduler]] | None = None) -> None:
        """Default scheduler getting object. Pass any scheduler functions through `schedulers` as a dictonary 
        of strings mapping to uninstantiated schedulers. `OneCycleLR` and `StepLR` are configured.

        Args:
            schedulers (Dict[str, Type[LRScheduler]] | None, optional): Dictonary mapping schedulers to their names. Defaults to None.
        """
        super().__init__()
        
        self.scheduler_registry: Dict[str, Type[LRScheduler]] = {
            'onecycle': OneCycleLR,
            'step': StepLR,
            
        }
        if schedulers is not None:
            self.scheduler_registry.update(schedulers)
        
    def __call__(
        self, 
        scheduler_name: str, 
        optim: Optimizer, 
        **kwargs
        ) -> LRScheduler:
        
        try:
            return self.scheduler_registry[scheduler_name](optim, **kwargs)
        except KeyError:
            raise NotImplementedError(f'{scheduler_name} is not a configured scheduler. Must be one of {list(self.scheduler_registry.keys())}')
        
    @property
    def configured_schedulers(self) -> list[str]:
        return list(self.scheduler_registry.keys())