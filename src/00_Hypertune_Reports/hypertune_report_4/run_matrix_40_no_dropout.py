from __future__ import annotations

import os
import time
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, Type

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.amp import autocast, GradScaler

from mads_datasets import DatasetFactoryProvider, DatasetType

# Use YOUR mltrainer API (from attachments)
from mltrainer import Trainer, TrainerSettings, ReportTypes
from mltrainer.metrics import Accuracy  # stateless metric: returns float per batch


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Speed-biased (close enough) reproducibility
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# ----------------------------
# Data
# ----------------------------
def get_streamers(batchsize: int):
    flowers_factory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
    # BasePreprocessor just stacks tensors (same as your attachment)
    from mltrainer.preprocessors import BasePreprocessor

    return flowers_factory.create_datastreamer(
        batchsize=batchsize,
        preprocessor=BasePreprocessor(),
    )


def get_device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Models (same core as before, compatible with your configs)
# ----------------------------
class CNNNoSkip(nn.Module):
    def __init__(
        self,
        filters: int,
        num_conv_blocks: int,
        hidden_sizes: list[int],
        input_size: Tuple[int, int, int, int] = (32, 3, 224, 224),
    ) -> None:
        super().__init__()
        in_channels = input_size[1]

        if not (1 <= num_conv_blocks <= 7):
            raise ValueError("num_conv_blocks must be between 1 and 7 for 224x224 inputs")

        first = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        others = [
            nn.Sequential(
                nn.Conv2d(filters, filters, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            for _ in range(num_conv_blocks - 1)
        ]
        self.blocks = nn.ModuleList([first] + others)

        # infer agg pool size
        with torch.no_grad():
            x = torch.ones(input_size)
            for b in self.blocks:
                x = b(x)
            h, w = x.shape[-2:]
        logger.info(f"[NoSkip] activation map -> ({h},{w})")
        self.agg = nn.AvgPool2d((h, w))

        self.flatten = nn.Flatten()
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()

        in_feats = filters
        for hsize in hidden_sizes:
            self.fcs.append(nn.Linear(in_feats, hsize))
            self.bns.append(nn.BatchNorm1d(hsize))
            in_feats = hsize

        #self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(in_feats, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        x = self.agg(x)
        x = self.flatten(x)
        for fc, bn in zip(self.fcs, self.bns):
            # x = self.dropout(torch.relu(bn(fc(x))))
            x = torch.relu(bn(fc(x)))
        return self.out(x)


class CNNSkip(nn.Module):
    def __init__(
        self,
        filters: int,
        num_conv_blocks: int,
        hidden_sizes: list[int],
        input_size: Tuple[int, int, int, int] = (32, 3, 224, 224),
    ) -> None:
        super().__init__()
        import torch.nn.functional as F  # noqa: F401

        in_channels = input_size[1]

        if not (1 <= num_conv_blocks <= 7):
            raise ValueError("num_conv_blocks must be between 1 and 7 for 224x224 inputs")

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(in_channels, filters, kernel_size=3, padding=1))
        for _ in range(num_conv_blocks - 1):
            self.convs.append(nn.Conv2d(filters, filters, kernel_size=3, padding=1))

        self.first_proj = nn.Conv2d(in_channels, filters, kernel_size=1) if in_channels != filters else nn.Identity()
        self.pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(num_conv_blocks)])

        final_spatial = 224 // (2 ** num_conv_blocks)
        logger.info(f"[Skip] activation map -> ({final_spatial},{final_spatial})")
        self.agg = nn.AvgPool2d((final_spatial, final_spatial))

        self.flatten = nn.Flatten()
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()

        in_feats = filters
        for hsize in hidden_sizes:
            self.fcs.append(nn.Linear(in_feats, hsize))
            self.bns.append(nn.BatchNorm1d(hsize))
            in_feats = hsize

        #self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(in_feats, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        # block 0
        identity = self.first_proj(x)
        x = self.convs[0](x) + identity
        x = F.relu(x)
        x = self.pools[0](x)

        for i in range(1, len(self.convs)):
            identity = x
            x = self.convs[i](x) + identity
            x = F.relu(x)
            x = self.pools[i](x)

        x = self.agg(x)
        x = self.flatten(x)
        for fc, bn in zip(self.fcs, self.bns):
            # x = self.dropout(F.relu(bn(fc(x))))
            x = F.relu(bn(fc(x)))
        return self.out(x)


# ----------------------------
# mltrainer-friendly AMP Trainer
# ----------------------------
class AmpTrainer(Trainer):
    """
    Uses mltrainer.Trainer for everything (settings, optimizer, scheduler, reporting),
    but swaps train/eval batches to use autocast + GradScaler on CUDA.
    Trainer's evalbatches() already averages metrics by summing batch floats. :contentReference[oaicite:5]{index=5}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_amp = (self.device == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)

        # tracking (for summary metrics)
        self.best_val_loss = float("inf")
        self.best_val_acc = -1.0
        self.best_epoch_val_loss = -1
        self.best_epoch_val_acc = -1
        self.best_state_dict: Optional[dict] = None

        self.final_val_loss = float("nan")
        self.final_val_acc = float("nan")

    def trainbatches(self) -> float:
        self.model.train()
        train_loss = 0.0
        it = iter(self.traindataloader)
        for _ in range(self.settings.train_steps):
            x, y = next(it)
            if self.device:
                x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()  # type: ignore

            if self.use_amp:
                with autocast(device_type="cuda"):
                    yhat = self.model(x)
                    loss = self.loss_fn(yhat, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)  # type: ignore
                self.scaler.update()
            else:
                yhat = self.model(x)
                loss = self.loss_fn(yhat, y)
                loss.backward()
                self.optimizer.step()  # type: ignore

            train_loss += float(loss.detach().cpu())

        return train_loss / self.settings.train_steps

    def evalbatches(self) -> Tuple[Dict[str, float], float]:
        self.model.eval()
        test_loss = 0.0
        metric_dict: Dict[str, float] = {}
        it = iter(self.validdataloader)

        with torch.no_grad():
            for _ in range(self.settings.valid_steps):
                x, y = next(it)
                if self.device:
                    x, y = x.to(self.device), y.to(self.device)

                if self.use_amp:
                    with autocast(device_type="cuda"):
                        yhat = self.model(x)
                        loss = self.loss_fn(yhat, y)
                else:
                    yhat = self.model(x)
                    loss = self.loss_fn(yhat, y)

                test_loss += float(loss.detach().cpu())

                # metrics are stateless and return float per batch :contentReference[oaicite:6]{index=6}
                for m in self.settings.metrics:
                    metric_dict[str(m)] = metric_dict.get(str(m), 0.0) + m(y, yhat)

        test_loss /= self.settings.valid_steps
        for k in list(metric_dict.keys()):
            metric_dict[k] = metric_dict[k] / self.settings.valid_steps

        # Scheduler behavior matches mltrainer.Trainer :contentReference[oaicite:7]{index=7}
        if self.scheduler:
            if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.scheduler.step(test_loss)
            else:
                self.scheduler.step()

        return metric_dict, test_loss

    def report(self, epoch: int, train_loss: float, test_loss: float, metric_dict: Dict) -> None:
        super().report(epoch, train_loss, test_loss, metric_dict)

        # Here "test_loss" is validation loss (it's the valid dataloader) :contentReference[oaicite:8]{index=8}
        val_acc = float(metric_dict.get("Accuracy", float("nan")))

        # Track best-by-val-loss and best-by-val-acc
        if test_loss < self.best_val_loss:
            self.best_val_loss = test_loss
            self.best_epoch_val_loss = epoch
            self.best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch_val_acc = epoch

        self.final_val_loss = test_loss
        self.final_val_acc = val_acc


# ----------------------------
# Evaluation on a chosen split (for "test_*")
# ----------------------------
@torch.no_grad()
def eval_split(
    model: nn.Module,
    dataloader: Iterator,
    steps: int,
    device: str,
    loss_fn: nn.Module,
    use_amp: bool,
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    metric = Accuracy()

    it = iter(dataloader)
    for _ in range(steps):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        if use_amp and device == "cuda":
            with autocast(device_type="cuda"):
                yhat = model(x)
                loss = loss_fn(yhat, y)
        else:
            yhat = model(x)
            loss = loss_fn(yhat, y)
        loss_sum += float(loss.detach().cpu())
        acc_sum += float(metric(y, yhat))  # float per batch

    return loss_sum / steps, acc_sum / steps


# ----------------------------
# Config + run list
# ----------------------------
@dataclass(frozen=True)
class RunConfig:
    batchsize: int
    train_steps: int
    valid_steps: int
    filters: int
    num_layers: int
    hidden: Tuple[int, int]
    skip: bool
    epochs: int
    seed: int

    @property
    def arch_key(self) -> str:
        return f"bs{self.batchsize}_ts{self.train_steps}_vs{self.valid_steps}_f{self.filters}_nl{self.num_layers}_h{self.hidden[0]}x{self.hidden[1]}"

    @property
    def run_name(self) -> str:
        return f"{self.arch_key}_{'skip' if self.skip else 'noskip'}_ep{self.epochs}_seed{self.seed}"


def build_runs() -> list[RunConfig]:
    seeds = [0, 1, 2, 3, 4]
    epochs_list = [15]
    skips = [False, True]

    archs = [
        # a
        dict(batchsize=32, train_steps=150, valid_steps=40, filters=96,  num_layers=5, hidden=(256, 128)),
        # b
        dict(batchsize=32, train_steps=150, valid_steps=40, filters=128, num_layers=5, hidden=(256, 128)),
        # c
        dict(batchsize=64, train_steps=75,  valid_steps=20, filters=96,  num_layers=5, hidden=(256, 128)),
        # d
        dict(batchsize=64, train_steps=75,  valid_steps=20, filters=96,  num_layers=4, hidden=(128, 128)),
    ]

    runs: list[RunConfig] = []
    for a in archs:
        for skip in skips:
            for ep in epochs_list:
                for sd in seeds:
                    runs.append(
                        RunConfig(
                            batchsize=a["batchsize"],
                            train_steps=a["train_steps"],
                            valid_steps=a["valid_steps"],
                            filters=a["filters"],
                            num_layers=a["num_layers"],
                            hidden=a["hidden"],
                            skip=skip,
                            epochs=ep,
                            seed=sd,
                        )
                    )
    assert len(runs) == 40
    return runs


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    # CUDA speed knobs (optional but usually helps on RTX-class GPUs)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "flowers_skip_matrix_80_mltrainer"))

    device = get_device_str()
    outdir = Path("matrix80_outputs_mltrainer")
    outdir.mkdir(parents=True, exist_ok=True)

    runs = build_runs()
    logger.info(f"Prepared {len(runs)} runs.")

    for i, cfg in enumerate(runs, start=1):
        set_seed(cfg.seed)
        logger.info(f"[{i:02d}/40] Starting {cfg.run_name}")

        streamers = get_streamers(cfg.batchsize)
        train_loader = streamers["train"].stream()
        valid_loader = streamers["valid"].stream()
        has_test = "test" in streamers
        test_loader = streamers["test"].stream() if has_test else streamers["valid"].stream()
        test_split_used = "test" if has_test else "valid"

        model_cls: Type[nn.Module] = CNNSkip if cfg.skip else CNNNoSkip
        model = model_cls(
            filters=cfg.filters,
            num_conv_blocks=cfg.num_layers,
            hidden_sizes=[cfg.hidden[0], cfg.hidden[1]],
            input_size=(cfg.batchsize, 3, 224, 224),
        )

        settings = TrainerSettings(
            epochs=cfg.epochs,
            metrics=[Accuracy()],                  # stateless batch metric :contentReference[oaicite:9]{index=9}
            logdir=outdir,                         # trainer will timestamp subdir :contentReference[oaicite:10]{index=10}
            train_steps=cfg.train_steps,
            valid_steps=cfg.valid_steps,
            reporttypes=[ReportTypes.MLFLOW],      # trainer logs per epoch to mlflow :contentReference[oaicite:11]{index=11}
            optimizer_kwargs={"lr": 1e-3, "weight_decay": 1e-5},
            scheduler_kwargs={"factor": 0.1, "patience": 10},
            earlystop_kwargs=None,
        )

        loss_fn = nn.CrossEntropyLoss()

        run_start = time.time()
        with mlflow.start_run(run_name=cfg.run_name):
            # tags/params for later grouping
            mlflow.set_tag("skip_layer", "Y" if cfg.skip else "N")
            mlflow.set_tag("regime", f"bs{cfg.batchsize}_ts{cfg.train_steps}_vs{cfg.valid_steps}")
            mlflow.set_tag("arch_key", cfg.arch_key)
            mlflow.set_tag("test_split_used", test_split_used)
            mlflow.log_params(
                {
                    "batchsize": cfg.batchsize,
                    "train_steps": cfg.train_steps,
                    "valid_steps": cfg.valid_steps,
                    "filters": cfg.filters,
                    "num_layers": cfg.num_layers,
                    "hidden": f"{cfg.hidden[0]}x{cfg.hidden[1]}",
                    "skip": cfg.skip,
                    "epochs": cfg.epochs,
                    "seed": cfg.seed,
                }
            )

            trainer = AmpTrainer(
                model=model,
                settings=settings,
                loss_fn=loss_fn,
                optimizer=optim.Adam,
                traindataloader=train_loader,
                validdataloader=valid_loader,
                scheduler=optim.lr_scheduler.ReduceLROnPlateau,
                device=device,
            )

            trainer.loop()

            # Evaluate FINAL on test split
            final_test_loss, final_test_acc = eval_split(
                model=trainer.model,
                dataloader=test_loader,
                steps=cfg.valid_steps,
                device=device,
                loss_fn=loss_fn,
                use_amp=(device == "cuda"),
            )

            # Evaluate BEST (by val loss) on test split
            best_test_loss = float("nan")
            best_test_acc = float("nan")
            if trainer.best_state_dict is not None:
                current = trainer.model.state_dict()
                trainer.model.load_state_dict(trainer.best_state_dict, strict=True)

                best_test_loss, best_test_acc = eval_split(
                    model=trainer.model,
                    dataloader=test_loader,
                    steps=cfg.valid_steps,
                    device=device,
                    loss_fn=loss_fn,
                    use_amp=(device == "cuda"),
                )

                trainer.model.load_state_dict(current, strict=True)

            # Summary metrics requested
            mlflow.log_metric("best_val_loss", trainer.best_val_loss)
            mlflow.log_metric("best_val_acc", trainer.best_val_acc)
            mlflow.log_metric("final_val_loss", trainer.final_val_loss)
            mlflow.log_metric("final_val_acc", trainer.final_val_acc)

            mlflow.log_metric("best_test_loss", best_test_loss)
            mlflow.log_metric("best_test_acc", best_test_acc)
            mlflow.log_metric("final_test_loss", final_test_loss)
            mlflow.log_metric("final_test_acc", final_test_acc)

            mlflow.log_param("best_epoch_val_loss", trainer.best_epoch_val_loss)
            mlflow.log_param("best_epoch_val_acc", trainer.best_epoch_val_acc)

            mlflow.log_metric("total_duration_sec", time.time() - run_start)

        logger.info(
            f"DONE {cfg.run_name} | "
            f"best_val_loss={trainer.best_val_loss:.4f} best_val_acc={trainer.best_val_acc:.4f} | "
            f"best_test_loss={best_test_loss:.4f} best_test_acc={best_test_acc:.4f}"
        )

    print("\nMLflow UI:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000")
    print("  http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
