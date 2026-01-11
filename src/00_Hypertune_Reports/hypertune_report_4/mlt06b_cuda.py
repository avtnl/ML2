from datetime import datetime
from pathlib import Path
from typing import Iterator

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from loguru import logger
from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer import TrainerSettings, metrics
from mltrainer.preprocessors import BasePreprocessor
from torch.amp import autocast, GradScaler  # Added for mixed precision


def get_flowers_datastreamers(batchsize: int):
    """Create the base datastreamer objects (without calling .stream() yet)."""
    flowers_factory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
    preprocessor = BasePreprocessor()
    streamers = flowers_factory.create_datastreamer(
        batchsize=batchsize,
        preprocessor=preprocessor,
    )
    return streamers


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


class CNN(nn.Module):
    # (unchanged – identical to original)
    def __init__(
        self,
        filters: int,
        num_conv_blocks: int,
        hidden_sizes: list[int],
        input_size: tuple[int, int, int, int] = (32, 3, 224, 224),
    ) -> None:
        super().__init__()
        self.in_channels = input_size[1]
        self.input_size = input_size
        self.filters = filters
        self.num_conv_blocks = num_conv_blocks
        self.hidden_sizes = hidden_sizes

        if not (1 <= num_conv_blocks <= 7):
            raise ValueError(
                f"num_conv_blocks must be between 1 and 7 for 224x224 inputs, got {num_conv_blocks}"
            )

        first_block = nn.Sequential(
            nn.Conv2d(self.in_channels, filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        other_blocks = [
            nn.Sequential(
                nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            )
            for _ in range(num_conv_blocks - 1)
        ]

        self.conv_blocks = nn.ModuleList([first_block] + other_blocks)

        activation_map_size = self._conv_test(input_size)
        logger.info(f"Aggregating activation map with size {activation_map_size}")
        self.agg = nn.AvgPool2d(activation_map_size)

        self.flatten = nn.Flatten()
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()

        in_features = filters
        for h in hidden_sizes:
            self.fcs.append(nn.Linear(in_features, h))
            self.bns.append(nn.BatchNorm1d(h))
            in_features = h

        self.out = nn.Linear(in_features, 5)
        self.dropout = nn.Dropout(p=0.2)

    def _apply_convs(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.conv_blocks:
            x = block(x)
        return x

    def _conv_test(self, input_size=(32, 3, 224, 224)) -> tuple[int, int]:
        x = torch.ones(input_size)
        x = self._apply_convs(x)
        return x.shape[-2:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_convs(x)
        x = self.agg(x)
        x = self.flatten(x)
        for fc, bn in zip(self.fcs, self.bns):
            x = fc(x)
            x = bn(x)
            x = torch.relu(x)
            x = self.dropout(x)
        logits = self.out(x)
        return logits


def setup_mlflow(experiment_path: str) -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_path)


def objective(params: dict) -> dict:
    modeldir = Path("models").resolve()
    if not modeldir.exists():
        modeldir.mkdir(parents=True)
        logger.info(f"Created model directory at {modeldir}")

    batchsize = 32
    logger.info(
        f"New trial with params: "
        f"filters={params['filters']}, "
        f"num_conv_blocks={params['num_conv_blocks']}, "
        f"hidden_sizes={params['hidden_sizes']}"
    )

    # Create base datastreamers (we will call .stream() fresh each epoch)
    streamers = get_flowers_datastreamers(batchsize)

    accuracy = metrics.Accuracy()

    settings = TrainerSettings(
        epochs=15,
        metrics=[accuracy],
        logdir=Path("modellog"),
        train_steps=150,
        valid_steps=40,
        reporttypes=[],  # we log manually now
    )

    device = get_device()

    # Clear CUDA cache at the start of each trial
    if device.type == "cuda":
        torch.cuda.empty_cache()

    with mlflow.start_run():
        mlflow.set_tag("model", "convnet_flowers_modulelist")
        mlflow.set_tag("dataset", "FLOWERS_5")
        mlflow.set_tag("script", "mlt06b")
        mlflow.set_tag("params", "bs32_ep15_max40_ts150vs40_amp")
        mlflow.set_tag("dev", "raoul_AvT")

        mlflow.log_params(params)
        mlflow.log_param("batchsize", batchsize)

        loss_fn = torch.nn.CrossEntropyLoss()
        model = CNN(**params).to(device)

        optimizer = optim.Adam(model.parameters())  # default lr=0.001
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scaler = GradScaler()

        test_loss = float("inf")

        for epoch in range(settings.epochs):
            # Fresh iterators each epoch → new shuffling for train, consistent valid
            trainstreamer = streamers["train"].stream()
            validstreamer = streamers["valid"].stream()

            # ------------------ Training ------------------
            model.train()
            train_loss = 0.0
            accuracy.reset()

            for _ in range(settings.train_steps):
                x, y = next(trainstreamer)
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()

                with autocast(device_type="cuda"):
                    logits = model(x)
                    loss = loss_fn(logits, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                accuracy(logits, y)  # assumes metric accepts logits

            avg_train_loss = train_loss / settings.train_steps
            train_acc = accuracy.compute().item()

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)

            # ------------------ Validation ------------------
            model.eval()
            valid_loss = 0.0
            accuracy.reset()

            with torch.no_grad():
                for _ in range(settings.valid_steps):
                    x, y = next(validstreamer)
                    x, y = x.to(device), y.to(device)

                    with autocast(device_type="cuda"):
                        logits = model(x)
                        loss = loss_fn(logits, y)

                    valid_loss += loss.item()
                    accuracy(logits, y)

            avg_valid_loss = valid_loss / settings.valid_steps
            valid_acc = accuracy.compute().item()

            mlflow.log_metric("valid_loss", avg_valid_loss, step=epoch)
            mlflow.log_metric("valid_accuracy", valid_acc, step=epoch)

            scheduler.step(avg_valid_loss)
            test_loss = avg_valid_loss  # final / last valid loss for hyperopt

        # Save model (after training)
        tag = datetime.now().strftime("%Y%m%d-%H%M")
        modelpath = modeldir / f"{tag}_flowers_modulelist_model.pt"
        logger.info(f"Saving model to {modelpath}")
        torch.save(model, modelpath)
        mlflow.log_artifact(local_path=str(modelpath), artifact_path="pytorch_models")

        return {"loss": test_loss, "status": STATUS_OK}


def main() -> None:
    setup_mlflow("mlflow_database_flowers")

    search_space = {
        "filters": scope.int(hp.quniform("filters", 32, 128, 32)),
        "num_conv_blocks": hp.choice("num_conv_blocks", [4, 5]),
        "hidden_sizes": hp.choice(
            "hidden_sizes",
            [
                [128, 128],
                [256, 128],
            ],
        ),
    }

    trials = Trials()
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=40,
        trials=trials,
    )

    logger.info(f"Best result: {best_result}")

    print("\n=== MLflow tracking info ===")
    print("Tracking URI : sqlite:///mlflow.db")
    print("Experiment   : 'mlflow_database_flowers'")
    print("\nTo open the MLflow UI, run this in your ML2 project directory:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000")
    print("Then open this URL in your browser:")
    print("  http://127.0.0.1:5000\n")


if __name__ == "__main__":
    main()