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
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics
from mltrainer.preprocessors import BasePreprocessor


def get_flowers_streamers(batchsize: int) -> tuple[Iterator, Iterator]:
    """Create train/valid streamers for the FLOWERS dataset."""
    flowers_factory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
    preprocessor = BasePreprocessor()
    streamers = flowers_factory.create_datastreamer(
        batchsize=batchsize,
        preprocessor=preprocessor,
    )
    train = streamers["train"]
    valid = streamers["valid"]
    trainstreamer = train.stream()
    validstreamer = valid.stream()
    return trainstreamer, validstreamer


def get_device() -> torch.device:
    """
    Select a single, consistent device object.

    GPU first (cuda) if available, otherwise CPU.
    No MPS / Apple support, per project preference.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


class CNN(nn.Module):
    """
    Simple ConvNet for FLOWERS (RGB, 224x224).

    - Expects 4D input tensors: (batch, channels=3, height=224, width=224)
    - Outputs logits for 5 flower classes.
    """

    def __init__(
        self,
        filters: int,
        units1: int,
        units2: int,
        input_size: tuple[int, int, int, int] = (32, 3, 224, 224),
    ) -> None:
        super().__init__()
        self.in_channels = input_size[1]
        self.input_size = input_size
        self.filters = filters
        self.units1 = units1
        self.units2 = units2

        self.convolutions = nn.Sequential(
            nn.Conv2d(self.in_channels, filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        activation_map_size = self._conv_test(input_size)
        logger.info(f"Aggregating activation map with size {activation_map_size}")
        self.agg = nn.AvgPool2d(activation_map_size)

        # self.dense = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(filters, units1),
        #     nn.ReLU(),
        #     nn.Linear(units1, units2),
        #     nn.BatchNorm2d(units1, units2),
        #     nn.ReLU(),
        #     nn.Dropout(0.20)
        #     # 5 flower classes instead of 10 FashionMNIST classes
        #     nn.Linear(units2, 5),
        # )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters, units1),
            nn.BatchNorm1d(units1),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(units1, units2),
            nn.BatchNorm1d(units2),
            nn.ReLU(),
            nn.Dropout(0.20),
            # 5 flower classes instead of 10 FashionMNIST classes
            nn.Linear(units2, 5),
        )


    def _conv_test(self, input_size: tuple[int, int, int, int]) -> tuple[int, int]:
        x = torch.ones(input_size)
        x = self.convolutions(x)
        return x.shape[-2:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutions(x)
        x = self.agg(x)
        logits = self.dense(x)
        return logits


def setup_mlflow(experiment_path: str) -> None:
    """
    Configure MLflow to use a SQLite backend and set the experiment name.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_path)


def objective(params: dict) -> dict:
    """
    Hyperopt objective function.

    - Creates datastreamers for FLOWERS
    - Builds a CNN with hyperparameters from `params`
    - Trains it with TrainerSettings + MLflow logging
    - Saves the model and returns validation loss for Hyperopt
    """
    modeldir = Path("models").resolve()
    if not modeldir.exists():
        modeldir.mkdir(parents=True)
        logger.info(f"Created model directory at {modeldir}")

    # Smaller batchsize than FashionMNIST, because images are larger (224x224x3)
    batchsize = 32
    trainstreamer, validstreamer = get_flowers_streamers(batchsize)
    accuracy = metrics.Accuracy()

    settings = TrainerSettings(
        epochs=15,
        metrics=[accuracy],
        logdir=Path("modellog"),
        train_steps=75,
        valid_steps=20,
        reporttypes=[ReportTypes.MLFLOW],
    )

    device = get_device()

    with mlflow.start_run():
        # Tags / metadata
        mlflow.set_tag("model", "convnet_flowers", "bigger NN")
        mlflow.set_tag("dataset", "FLOWERS")
        mlflow.set_tag("dev", "raoul")

        # Log hyperparameters
        mlflow.log_params(params)
        mlflow.log_param("batchsize", batchsize)

        # Optimizer / loss
        optimizer_cls = optim.Adam
        loss_fn = torch.nn.CrossEntropyLoss()

        # Instantiate model with current hyperparameters and move to device
        model = CNN(**params).to(device)

        # Train with Trainer
        trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=loss_fn,
            optimizer=optimizer_cls,  # type: ignore[arg-type]
            traindataloader=trainstreamer,
            validdataloader=validstreamer,
            scheduler=optim.lr_scheduler.ReduceLROnPlateau,
            device=device,  # same torch.device as model
        )
        trainer.loop()

        # Save the trained model
        tag = datetime.now().strftime("%Y%m%d-%H%M")
        modelpath = modeldir / f"{tag}_flowers_model.pt"
        logger.info(f"Saving model to {modelpath}")
        torch.save(model, modelpath)

        # Log as MLflow artifact
        mlflow.log_artifact(local_path=str(modelpath), artifact_path="pytorch_models")

        # Hyperopt minimizes this loss
        return {"loss": trainer.test_loss, "status": STATUS_OK}


def main() -> None:
    setup_mlflow("mlflow_database_flowers")

    # Hyperparameter search space: only "safe" parameters (no geometry changes)
    # search_space = {
    #     "filters": scope.int(hp.quniform("filters", 16, 128, 16)),
    #     "units1": scope.int(hp.quniform("units1", 32, 128, 32)),
    #     "units2": scope.int(hp.quniform("units2", 32, 128, 32)),
    # }
    search_space = {
        "filters": scope.int(hp.quniform("filters", 64, 128, 64)),
        "units1": scope.int(hp.quniform("units1", 64, 128, 64)),
        "units2": scope.int(hp.quniform("units2", 64, 128, 64)),
    }
    trials = Trials()
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=8,
        trials=trials,
    )

    logger.info(f"Best result: {best_result}")

    # Helpful hint on how to open MLflow UI for this script
    print("\n=== MLflow tracking info ===")
    print("Tracking URI : sqlite:///mlflow.db")
    print("Experiment   : 'mlflow_database_flowers'")
    print("\nTo open the MLflow UI, run this in your ML2 project directory:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000")
    print("Then open this URL in your browser:")
    print("  http://127.0.0.1:5000\n")


if __name__ == "__main__":
    main()
