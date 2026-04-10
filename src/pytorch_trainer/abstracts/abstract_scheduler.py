from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from torch.optim.optimizer import Optimizer
from abc import ABC, abstractmethod


class SchedulerGetter(ABC):
    
    def __init__(self) -> None:
        
        super().__init__()
    
    
    @abstractmethod
    def __call__(
        self, 
        scheduler_name: str, 
        optim: Optimizer, 
        **kwargs
        ) -> LRScheduler:
        """Overwrite this method to return your configured scheduler.

        Args:
            scheduler_name (str): Name of scheduler.
            optim (Optimizer): Preconfigured optimizer.

        Raises:
            NotImplementedError: If scheduler name is not in your configured schedulers.

        Returns:
            LRScheduler: Configured scheduler.
        """
        configured_schedulers = [
                'cosine'
        ]
            
        if scheduler_name not in configured_schedulers:
            raise NotImplementedError(f'{scheduler_name} is not a configured scheduler, `scheduler_name` must be one of {configured_schedulers}')
        
        if scheduler_name == 'cosine':
            
            return OneCycleLR(
                optimizer=optim,
                **kwargs
            )
            
        
