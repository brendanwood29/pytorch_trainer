from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary
from tqdm import tqdm

from .config import Config
from .defaults import (
    EarlyStopping,
    LossGetter,
    ModelGetter,
    OptimGetter,
    SchedulerGetter,
)


class Trainer(ABC):
    def __init__(
        self,
        cfg_path: str | Path,
        get_model: ModelGetter,
        get_optim: OptimGetter = OptimGetter(),
        get_scheduler: SchedulerGetter = SchedulerGetter(),
        get_loss_fn: LossGetter = LossGetter(),
    ) -> None:

        cfg = self.validate_config(cfg_path)

        self.model = get_model(cfg.model.name, **cfg.model.kwargs)
        self.optimizer = get_optim(
            cfg.optim.name,
            model_params=self.model.parameters(),
            lr=cfg.optim.lr,
            **cfg.optim.kwargs,
        )
        self.loss_fn = get_loss_fn(cfg.loss.name, **cfg.loss.kwargs)
        self.scheduler = None
        if cfg.scheduler is not None:
            self.lr_history = []
            self.scheduler = get_scheduler(
                cfg.scheduler.name, optim=self.optimizer, **cfg.scheduler.kwargs
            )

        if cfg.model.init:
            self.load_model(
                cfg.model.init.weights,
                weights_only=cfg.model.init.weights_only,
                strict=cfg.model.init.strict,
            )

        if cfg.model.freeze_modules:
            self.edit_requires_grad(cfg.model.freeze_modules, requires_grad=False)
        if cfg.model.unfreeze_modules:
            self.edit_requires_grad(cfg.model.unfreeze_modules, requires_grad=True)
        if cfg.model.print_summary:
            summary(self.model)

        self.stopper = None
        if cfg.early_stopping:
            self.stopper = EarlyStopping(
                cfg.early_stopping.patience, cfg.early_stopping.threshold
            )

        self.cfg = cfg
        self.device: str | torch.Device = self.cfg.device
        self.use_ddp = False
        self.loss_epoch: List[float] = []
        self.val_loss: List[float] = []
        self.step_loss: List[float] = []
        self.run_name: str = cfg.run_name
        self.work_dir = Path(cfg.work_dir).joinpath(self.run_name)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.step: int = 0
        self.best_val_loss: float = torch.inf
        self.top_saved_models: List[Tuple[float, Path]] = []
        self.num_models_save: int = cfg.model.num_models_to_save

        if "cuda:" in cfg.device:
            self.model.to(cfg.device)
        elif cfg.device == "all":
            self.use_ddp = True
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(dist.get_rank())
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = torch.device(f"cuda:{self.rank}")
            self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[self.rank])
        else:
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        OmegaConf.save(self.cfg, self.work_dir.joinpath("config.yaml"))

    def validate_config(
        self, cfg: str | Path | DictConfig | ListConfig
    ) -> DictConfig | ListConfig:
        schema = OmegaConf.structured(Config)
        OmegaConf.set_struct(schema, False)
        if isinstance(cfg, str) or isinstance(cfg, Path):
            cfg = OmegaConf.load(cfg)
        cfg = OmegaConf.merge(schema, cfg)
        OmegaConf.to_container(cfg, throw_on_missing=True)

        if cfg.scheduler is not None:
            OmegaConf.to_container(cfg.scheduler, throw_on_missing=True)
        if cfg.early_stopping is not None:
            OmegaConf.to_container(cfg.early_stopping, throw_on_missing=True)
        if cfg.grad_clip is not None:
            OmegaConf.to_container(cfg.grad_clip, throw_on_missing=True)

        return cfg

    def __call__(self, train_dataset: Dataset, val_dataset: Dataset) -> None:

        self.train_loader = self.configure_dataloader(
            train_dataset, self.cfg.batch_size, self.cfg.data.train.shuffle
        )
        self.val_loader = self.configure_dataloader(
            val_dataset, self.cfg.batch_size, self.cfg.data.val.suffle
        )

        with tqdm(range(self.cfg.num_epochs), leave=False) as pbar:
            for final_model_epochs in pbar:
                self.current_epoch = final_model_epochs
                self.train(self.train_loader)
                pbar.set_postfix(
                    {
                        "train_loss": f"{self.loss_epoch[-1]:.4f}",
                        "val_loss": f"{self.last_val_loss:.4f}",
                    },
                    refresh=False,
                )
                with torch.no_grad():
                    should_stop = self.val(self.val_loader)
                if should_stop:
                    print(
                        (
                            f"Stopped after {self.current_epoch}"
                            "epochs due to early stopping."
                        )
                    )
                    break
        dist.destroy_process_group()
        self.after_training()
        self.training_summary(self.current_epoch, save_final=self.cfg.model.save_last)

    def configure_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool):

        if self.use_ddp:
            sampler = DistributedSampler(
                dataset=dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                seed=self.cfg.seed,
            )
            loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)
        else:
            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
            )

        return loader

    @abstractmethod
    def model_forward(self, batch) -> Tuple[torch.Tensor, int]:
        """Abstract method for a forward pass of a model

        Args:
            batch: Batch from DataLoader.
        Returns:
            (loss, current_batch_size)
        """
        pass

    def after_training(self):
        """This method runs after training is finished.
        Overwrite it with an evaluation script.
        """
        pass

    def clip_grad_norm(self) -> None:
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.cfg.grad_clip.max_norm
        )

    def after_train_batch(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.grad_clip is not None:
            self.clip_grad_norm()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
            self.lr_history.append(self.scheduler.get_last_lr())

    def train(self, train_loader: DataLoader) -> None:
        loss_iters = 0
        samples_processed = 0
        self.model.train()
        with tqdm(train_loader, disable=not self.cfg.batch_pbar) as pbar:
            for x in pbar:
                x = [item.to(self.device) for item in x]
                pbar.set_description("Training Loop: ")
                loss, batch_size = self.model_forward(x)  # type: ignore
                loss_iters += loss.item() * batch_size
                samples_processed += batch_size
                pbar.set_postfix({"train_loss_step": f"{loss.item():.4f}"})
                self.step_loss.append(loss.item())
                self.step += 1
                self.after_train_batch(loss)
        self.loss_epoch.append(loss_iters / samples_processed)

    def val(self, val_loader: DataLoader) -> bool:
        self.model.eval()
        loss_iters = 0
        samples_processed = 0
        with tqdm(val_loader, disable=not self.cfg.batch_pbar) as pbar:
            pbar.set_description("Validation Loop: ")
            for x in pbar:
                x = [item.to(self.device) for item in x]
                loss, batch_size = self.model_forward(x)  # type: ignore
                loss_iters += loss.item() * batch_size
                samples_processed += batch_size
                pbar.set_postfix({"val_loss_step": f"{loss.item():.4f}"})
        self.val_loss.append(loss_iters / samples_processed)

        if self.last_val_loss < self.best_val_loss:
            self.best_val_loss = self.last_val_loss
            self.save_after_val()

        if self.stopper is not None:
            return self.stopper(self.last_val_loss)
        return False

    @property
    def last_val_loss(self):
        return self.val_loss[-1] if self.val_loss else float(torch.inf)

    def save_after_val(self):

        model_name = self.save_model()

        self.top_saved_models.append((self.last_val_loss, model_name))
        self.top_saved_models.sort(key=lambda x: x[0])
        if len(self.top_saved_models) > self.num_models_save:
            _, worst_path = self.top_saved_models.pop()
            if worst_path.is_file():
                worst_path.unlink()

    def load_model(
        self, file: str | Path, weights_only: bool, strict: bool = True
    ) -> None:

        pt_file = torch.load(file)

        self.model.load_state_dict(pt_file["model_state"], strict=strict)
        if weights_only:
            return
        self.optimizer.load_state_dict(pt_file["optim_state"])
        self.val_loss.append(pt_file["val_loss"])
        if "scheduler_state" in pt_file:
            self.scheduler.load_state_dict(pt_file["scheduler_state"])  # type: ignore

    def save_model(self, model_name: str = "model.pt") -> Path:
        out_dir = self.work_dir.joinpath("models")
        out_dir.mkdir(parents=True, exist_ok=True)
        params = {
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "val_loss": self.last_val_loss,
        }
        if self.scheduler is not None:
            params["scheduler_state"] = self.scheduler.state_dict()
        if model_name == "model.pt":
            model_name = (
                f"{self.run_name}-epoch-{self.current_epoch}"
                f"_best_val_loss_{self.last_val_loss:.4f}.pt"
            )
        model_path = out_dir.joinpath(model_name)
        torch.save(params, model_path)

        return model_path

    def edit_requires_grad(
        self, modules: list, requires_grad: bool, verbose: bool = True
    ):
        params_changed = 0
        if requires_grad:
            action = "Unfreezing"
        else:
            action = "Freezing"

        for name, param in self.model.named_parameters():
            for keyword in modules:
                if isinstance(keyword, list):
                    # Check if all specify keywords are in param name,
                    # if keyword is list
                    if all([word in name for word in keyword]):
                        if verbose:
                            print(f"{action} {name}")
                        param.requires_grad = requires_grad
                        params_changed += torch.numel(param.data)

                else:
                    if keyword in name:
                        if verbose:
                            print(f"{action} {name}")
                        param.requires_grad = requires_grad
                        params_changed += torch.numel(param.data)

    def training_summary(self, final_epochs: int, save_final: bool):
        if save_final:
            self.save_model(model_name="final_model.pt")

        fig_dir = self.work_dir.joinpath("figures")
        fig_dir.mkdir(parents=True, exist_ok=True)

        print(
            (
                "\nModel finished training with best validation"
                f"{self.cfg.loss.name}: {self.best_val_loss:.4f}"
            )
        )

        plt.figure()
        plt.plot(torch.linspace(1, self.step, final_epochs + 1), self.loss_epoch, "r")
        plt.plot(torch.linspace(1, self.step, final_epochs + 1), self.val_loss)
        plt.xlabel("Steps")
        plt.ylabel(f"{self.cfg.loss.name}")
        plt.legend(["Train Loss", "Validation Loss"])
        plt.tight_layout()
        plt.savefig(fig_dir.joinpath("loss_curves.png"))
        plt.close()

        if self.scheduler is not None:
            plt.figure()
            plt.plot(range(self.step), self.lr_history)
            plt.xlabel("Step")
            plt.ylabel("Learning Rate")
            plt.tight_layout()
            plt.savefig(fig_dir.joinpath("lr.png"))
            plt.close()

        plt.figure()
        plt.plot(range(self.step), self.step_loss)
        plt.xlabel("Step")
        plt.ylabel(f"{self.cfg.loss.name}")
        plt.tight_layout()
        plt.savefig(fig_dir.joinpath("step_loss.png"))
        plt.close()
