# mlt02.py - Hyperparameter sweep with mltrainer + MLflow (HU ML2 style - same structure as mlt01a/mlt01b)

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
import toml
from pathlib import Path

# MLflow for experiment tracking
import mlflow

# Exactly the same imports as in mlt01a and mlt01b
from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer.preprocessors import BasePreprocessor
from mltrainer import metrics, TrainerSettings, ReportTypes, Trainer

# ==================== DEVICE ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA IS WORKING! Using GPU → {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CUDA not available → using CPU")
logger.info(f"Using device: {device}")

# ==================== DATA ====================
# Prepare dataset using mads_datasets datastreamer (fast and clean)
fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)

streamers = fashionfactory.create_datastreamer(
    batchsize=64,
    preprocessor=BasePreprocessor()
)

train_streamer = streamers["train"].stream()
valid_streamer = streamers["valid"].stream()

print(f"Train batches: {len(streamers['train'])} | Valid batches: {len(streamers['valid'])}")

# ==================== MODEL ====================
# Neural Network (same architecture as used in previous assignments)
class NeuralNetwork(nn.Module):
    def __init__(self, num_classes: int = 10, units1: int = 512, units2: int = 256):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, units1),
            nn.BatchNorm1d(units1),
            nn.ReLU(),
            #  nn.Dropout(0.15),
            nn.Linear(units1, units2),
            nn.BatchNorm1d(units2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(units2, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

# ==================== SWEEP CONFIGS ====================
sweep_configs = [
    {"units1": 256, "units2": 128, "epochs": 12},
    {"units1": 128, "units2": 128, "epochs": 12},
]

# ==================== MAIN SWEEP LOOP ====================
if __name__ == "__main__":

    log_dir = Path("log_mlt02")
    log_dir.mkdir(exist_ok=True)

    # MLflow experiment setup
    mlflow.set_experiment("ML2 - FashionMNIST Hyperparameter Sweep")

    for idx, config in enumerate(sweep_configs, 1):
        # --- derive hyperparams with defaults ---
        DEFAULT_LR = 1e-3
        DEFAULT_WD = 0.0  # weight_decay
        lr = config.get("lr", DEFAULT_LR)
        wd = config.get("weight_decay", DEFAULT_WD)

        # run name uses lr & wd variables, never config["lr"] directly
        run_name = (
            f"run{idx:02d}_u1{config['units1']}_u2{config['units2']}_"
            f"lr{lr}_wd{wd}_ep{config['epochs']}"
        )

        with mlflow.start_run(run_name=run_name):
            logger.info(f"Starting {run_name}")
            mlflow.log_params(config)

            model = NeuralNetwork(
                num_classes=10,
                units1=config["units1"],
                units2=config["units2"],
            ).to(device)

            settings = TrainerSettings(
                epochs=config["epochs"],
                metrics=[metrics.Accuracy()],
                logdir=log_dir,
                train_steps=len(streamers["train"]),
                valid_steps=len(streamers["valid"]),
                reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.TOML],
                optimizer_kwargs={
                    "lr": lr,
                    "weight_decay": wd,
                },
            )

            trainer = Trainer(
                model=model,
                settings=settings,
                loss_fn=nn.CrossEntropyLoss(),
                optimizer=optim.Adam,
                traindataloader=train_streamer,
                validdataloader=valid_streamer,
                scheduler=optim.lr_scheduler.ReduceLROnPlateau,
                device=device,
            )

            trainer.loop()

            # Find the most recently created folder inside log_mlt02 (this is always the current run)
            log_base = Path(settings.logdir)
            timestamp_folder = max(log_base.iterdir(), key=lambda f: f.stat().st_mtime)

            metrics_path = timestamp_folder / "metrics.toml"

            if metrics_path.exists():
                data = toml.load(metrics_path)
                best_acc = float(data["best_metric"])
                mlflow.log_metric("best_valid_accuracy", best_acc)
                logger.info(f"Run finished → Best validation accuracy = {best_acc:.4f}")
            else:
                logger.warning("metrics.toml not found – this shouldn't happen with TOML enabled")

            # # Extract best validation accuracy from history and log to MLflow
            # if trainer.history:
            #     best_valid_acc = max(
            #         step["valid_metric"] for step in trainer.history if "valid_metric" in step
            #     )
            #     mlflow.log_metric("best_valid_accuracy", best_valid_acc)
            #     logger.info(f"Finished {run_name} → Best validation accuracy: {best_valid_acc:.4f}")
            # else:
            #     logger.warning("No history found – could not log best metric")

    print("\nSweep completed!")
    print("View results with:")
    print("   → TensorBoard: tensorboard --logdir log_mlt02")
    print("   → MLflow UI:    mlflow ui")