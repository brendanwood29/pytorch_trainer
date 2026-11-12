from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Dict, Any, List, Optional


@dataclass
class EarlyStoppingConfig:
    patience: int = MISSING
    threshold: float = MISSING

@dataclass
class ModelInitConfig:
    weights: str = MISSING
    strict: bool = MISSING
    weights_only: bool = MISSING


@dataclass
class ModelConfig:
    name: str = MISSING
    init: Optional[ModelInitConfig] = None
    num_models_to_save: int = MISSING
    print_summary: bool = MISSING
    save_last: bool = MISSING
    freeze_modules: Optional[List[str]] = field(default_factory=list)
    unfreeze_modules: Optional[List[str]] = field(default_factory=list)
    kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    

@dataclass
class GradClipConfig:
    max_norm: float = MISSING


@dataclass
class LossConfig:
    name: str = MISSING
    kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

    
@dataclass
class OptimConfig:
    name: str = MISSING 
    lr: float = MISSING
    kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    name: str = MISSING
    kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class Config:
    num_epochs: int = MISSING
    run_name: str = MISSING
    work_dir: str = MISSING
    optim: OptimConfig = MISSING
    loss: LossConfig = MISSING
    model: ModelConfig = MISSING
    device: str = 'cuda'
    batch_pbar: bool = MISSING
    scheduler: Optional[SchedulerConfig] = None
    grad_clip: Optional[GradClipConfig] = None
    early_stopping: Optional[EarlyStoppingConfig] = None
    