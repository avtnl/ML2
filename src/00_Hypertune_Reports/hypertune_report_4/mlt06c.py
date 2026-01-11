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
    CNN using ModuleList for both:
    - a configurable number of convolution blocks
    - a configurable dense head with variable depth

    Adapted for FLOWERS:
    - Expects input (batch, 3, 224, 224)
    - Outputs logits for 5 flower classes
    """

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

        # For 224x224 RGB images, up to several blocks is safe.
        # Our search space uses 1–3 blocks, which is comfortably safe.
        if not (1 <= num_conv_blocks <= 7):
            raise ValueError(
                f"num_conv_blocks must be between 1 and 7 for 224x224 inputs, got {num_conv_blocks}"
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
                nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
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
        self.bns = nn.ModuleList()

        # After global avg pooling: (batch, filters, 1, 1) -> (batch, filters)
        in_features = filters
        for h in hidden_sizes:
            self.fcs.append(nn.Linear(in_features, h))
            self.bns.append(nn.BatchNorm1d(h))
            in_features = h

        # Final classification layer for 5 FLOWERS classes
        self.out = nn.Linear(in_features, 5)
        self.dropout = nn.Dropout(p=0.2)

    def _apply_convs(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all convolutional blocks in sequence."""
        for block in self.conv_blocks:
            x = block(x)
        return x

    def _conv_test(self, input_size=(32, 3, 224, 224)) -> tuple[int, int]:
        """Run a dummy tensor through conv stack to find spatial output size."""
        x = torch.ones(input_size)
        x = self._apply_convs(x)
        return x.shape[-2:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_convs(x)
        x = self.agg(x)        # global average pool over HxW
        x = self.flatten(x)    # (batch, filters)
        # for fc in self.fcs:
        #     x = torch.relu(fc(x))
        for fc, bn in zip(self.fcs, self.bns):
            x = fc(x)
            x = bn(x)          # BatchNorm1d over features
            x = torch.relu(x)
            x = self.dropout(x)   
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
    batchsize = 64
    logger.info(
    f"New trial with params: "
    f"filters={params['filters']}, "
    f"num_conv_blocks={params['num_conv_blocks']}, "
    f"hidden_sizes={params['hidden_sizes']}"
    )
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
        mlflow.set_tag("model", "convnet_flowers_modulelist")
        mlflow.set_tag("dataset", "FLOWERS_5")
        mlflow.set_tag("script", "mlt06c")
        mlflow.set_tag("params", "bs64_ep15_max20_ts75vs20")
        mlflow.set_tag("dev", "raoul_AvT")

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
        modelpath = modeldir / f"{tag}_flowers_modulelist_model.pt"
        logger.info(f"Saving model to {modelpath}")
        torch.save(model, modelpath)

        # Log as MLflow artifact
        mlflow.log_artifact(local_path=str(modelpath), artifact_path="pytorch_models")

        # Hyperopt minimizes this loss
        return {"loss": trainer.test_loss, "status": STATUS_OK}


def main() -> None:
    setup_mlflow("mlflow_database_flowers")

    # Hyperparameter search space:
    # - filters: number of conv filters
    # - num_conv_blocks: how many conv blocks (1–3)
    # - hidden_sizes: structure of the dense head (list of layer widths)
    search_space = {
        "filters": scope.int(hp.quniform("filters", 32, 128, 32)),
        #"filters": hp.choice("filters", [32, 64, 128]),
        "num_conv_blocks": hp.choice("num_conv_blocks", [4, 5]),
        #"num_conv_blocks": scope.int(hp.quniform("num_conv_blocks", 4, 5, 1)),
        "hidden_sizes": hp.choice(
            "hidden_sizes",
            [
                # [64],
                # [64, 64],
                # [128, 64],
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
        max_evals=20,
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
