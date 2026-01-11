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


def get_fashion_streamers(batchsize: int) -> tuple[Iterator, Iterator]:
    """Create train/valid streamers for the FashionMNIST dataset."""
    fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
    preprocessor = BasePreprocessor()
    streamers = fashionfactory.create_datastreamer(
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


# There are more models in mltrainer.imagemodels for inspiration.
# You can import them, or create your own like here.
class CNN(nn.Module):
    """
    CNN using ModuleList for both:
    - a configurable number of convolution blocks
    - a configurable dense head with variable depth
    """

    def __init__(
        self,
        filters: int,
        num_conv_blocks: int,
        hidden_sizes: list[int],
        input_size: tuple[int, int, int, int] = (32, 1, 28, 28),
    ) -> None:
        super().__init__()
        self.in_channels = input_size[1]
        self.input_size = input_size
        self.filters = filters
        self.num_conv_blocks = num_conv_blocks
        self.hidden_sizes = hidden_sizes

        # For the fashion: current conv / pool geometry and 28x28 inputs, up to 3 blocks is safe:
        # 1 block: 28x28 -> 14x14
        # 2 blocks: 14x14 -> 12x12 -> 6x6
        # 3 blocks: 6x6 -> 4x4 -> 2x2
        # 4th block would collapse the spatial dimensions.
        if not (1 <= num_conv_blocks <= 3):
            raise ValueError(
                f"num_conv_blocks must be between 1 and 3 for 28x28 inputs, got {num_conv_blocks}"
            )

        # ----- CONVOLUTIONAL PART: dynamic depth with ModuleList -----
        # First block: in_channels -> filters, keep same spatial size then downsample
        first_block = nn.Sequential(
            nn.Conv2d(self.in_channels, filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Subsequent blocks: filters -> filters, shrink spatial size, then downsample
        other_blocks = [
            nn.Sequential(
                nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            )
            for _ in range(num_conv_blocks - 1)
        ]

        self.conv_blocks = nn.ModuleList([first_block] + other_blocks)

        # Infer activation map size by applying the same conv stack to a dummy tensor
        activation_map_size = self._conv_test(input_size)
        logger.info(f"Aggregating activation map with size {activation_map_size}")
        self.agg = nn.AvgPool2d(activation_map_size)

        # ----- DENSE PART: dynamic head with ModuleList -----
        self.flatten = nn.Flatten()
        self.fcs = nn.ModuleList()

        in_features = filters  # after global avg pooling: (batch, filters, 1, 1) -> (batch, filters)
        for h in hidden_sizes:
            self.fcs.append(nn.Linear(in_features, h))
            in_features = h

        # Final classification layer for 10 FashionMNIST classes
        self.out = nn.Linear(in_features, 10)

    def _apply_convs(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all convolutional blocks in sequence."""
        for block in self.conv_blocks:
            x = block(x)
        return x

    def _conv_test(self, input_size=(32, 1, 28, 28)) -> tuple[int, int]:
        """Run a dummy tensor through conv stack to find spatial output size."""
        x = torch.ones(input_size)
        x = self._apply_convs(x)
        return x.shape[-2:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_convs(x)
        x = self.agg(x)        # global average pool over HxW
        x = self.flatten(x)    # (batch, filters)
        for fc in self.fcs:
            x = torch.relu(fc(x))
        logits = self.out(x)
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

    - Creates datastreamers for FashionMNIST
    - Builds a CNN with hyperparameters from `params`
    - Trains it with TrainerSettings + MLflow logging
    - Saves the model and returns validation loss for Hyperopt
    """
    modeldir = Path("models").resolve()
    if not modeldir.exists():
        modeldir.mkdir(parents=True)
        logger.info(f"Created model directory at {modeldir}")

    batchsize = 32
    trainstreamer, validstreamer = get_fashion_streamers(batchsize)
    accuracy = metrics.Accuracy()

    settings = TrainerSettings(
        epochs=15,
        metrics=[accuracy],
        logdir=Path("modellog"),
        train_steps=150,
        valid_steps=40,
        reporttypes=[ReportTypes.MLFLOW],
    )

    device = get_device()

    with mlflow.start_run():
        # Tags / metadata
        mlflow.set_tag("flowers5-mlt05b", "bs32_ep14_max40_trstp150_valstp_40")
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
        modelpath = modeldir / f"{tag}_model.pt"
        logger.info(f"Saving model to {modelpath}")
        torch.save(model, modelpath)

        # Log as MLflow artifact
        mlflow.log_artifact(local_path=str(modelpath), artifact_path="pytorch_models")

        # Hyperopt minimizes this loss
        return {"loss": trainer.test_loss, "status": STATUS_OK}


def main() -> None:
    setup_mlflow("mlflow_database")

    # Hyperparameter search space:
    # - filters: number of conv filters
    # - num_conv_blocks: how many conv blocks (1–3; geometry-safe for 28x28 input)
    # - hidden_sizes: structure of the dense head (list of layer widths)
    search_space = {
        "filters": scope.int(hp.quniform("filters", 32, 128, 32)),
        #"filters": hp.choice("filters", [32, 64, 128]),
        "num_conv_blocks": hp.choice("num_conv_blocks", [4, 5]),
        #"num_conv_blocks": scope.int(hp.quniform("num_conv_blocks", 4, 5, 1)),
        "hidden_sizes": hp.choice(
            "hidden_sizes",
            [
                #[64],
                #[64, 64],
                #[128, 64],
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

    # Helpful hint on how to open MLflow UI for this script
    print("\n=== MLflow tracking info ===")
    print("Tracking URI : sqlite:///mlflow.db")
    print("Experiment   : 'mlflow_database'")
    print("\nTo open the MLflow UI, run this in your ML2 project directory:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000")
    print("Then open this URL in your browser:")
    print("  http://127.0.0.1:5000\n")


if __name__ == "__main__":
    main()
