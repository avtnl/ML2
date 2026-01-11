from pathlib import Path
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer.preprocessors import BasePreprocessor
from mltrainer import metrics, Trainer, TrainerSettings, ReportTypes
from torchinfo import summary

import mlflow
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope

# ---------------------------------------------------------------------
# General setup
# ---------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)

# Where to store models / logs
modeldir = Path("models").resolve()
modeldir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# Data: Fashion-MNIST
# ---------------------------------------------------------------------
fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
batchsize = 64
preprocessor = BasePreprocessor()
streamers = fashionfactory.create_datastreamer(
    batchsize=batchsize, preprocessor=preprocessor
)
train = streamers["train"]
valid = streamers["valid"]
trainstreamer = train.stream()
validstreamer = valid.stream()

# Grab one batch for sanity checks / summary input size
x, y = next(iter(trainstreamer))

# ---------------------------------------------------------------------
# Device (GPU first, then CPU)
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
logger.info(f"Using device: {device}")

# ---------------------------------------------------------------------
# Model definition: custom CNN with two dense layers
# ---------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, filters, units1, units2, input_size=(32, 1, 28, 28)):
        super().__init__()
        self.in_channels = input_size[1]
        self.input_size = input_size
        self.filters = filters
        self.units1 = units1
        self.units2 = units2

        # 3 conv blocks: Conv -> ReLU -> MaxPool(2)
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
        )

        # Work out the activation map size after convolutions
        activation_map_size = self._conv_test(input_size)
        logger.info(f"Aggregating activation map with size {activation_map_size}")
        # Global avg pool over spatial dims
        self.agg = nn.AvgPool2d(activation_map_size)

        # Dense block, using (units1, units2)
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters, units1),
            nn.BatchNorm1d(units1),
            nn.ReLU(),
            nn.Linear(units1, units2),
            nn.BatchNorm1d(units2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(units2, 10),  # 10 Fashion-MNIST classes
        )

    def _conv_test(self, input_size=(32, 1, 28, 28)):
        x = torch.ones(input_size)
        x = self.convolutions(x)
        return x.shape[-2:]  # (H, W)

    def forward(self, x):
        x = self.convolutions(x)
        x = self.agg(x)
        logits = self.dense(x)
        return logits


# ---------------------------------------------------------------------
# Default model + sanity check
# ---------------------------------------------------------------------
# Default choice: 32 filters, (units1, units2) = (64, 64)
default_model = CNN(filters=32, units1=64, units2=64).to(device)
summary(default_model, input_size=(32, 1, 28, 28))

# Quick sanity check on one batch
accuracy = metrics.Accuracy()
x = x.to(device)
y = y.to(device)
with torch.no_grad():
    yhat = default_model(x)
print("Sanity accuracy on one batch:", float(accuracy(y, yhat)))

# ---------------------------------------------------------------------
# Trainer settings
# ---------------------------------------------------------------------
settings = TrainerSettings(
    epochs=20,
    metrics=[accuracy],
    logdir=modeldir,  # logs + artifacts will land under ./models
    train_steps=100,
    valid_steps=100,
    reporttypes=[ReportTypes.MLFLOW, ReportTypes.TOML],
)

optimizer_cls = optim.Adam
loss_fn = torch.nn.CrossEntropyLoss()

# ---------------------------------------------------------------------
# Objective function for Hyperopt
# ---------------------------------------------------------------------
def objective(params):
    """
    Hyperopt objective:
      - tunes the number of convolution filters (filters)
      - tunes the size of the two dense layers (units1, units2)
      - uses the custom CNN class defined above
    """
    # Hyperopt suggestions (make sure they are ints)
    filters = int(params["filters"])
    dense_cfg = params["dense_config"]
    units1 = int(dense_cfg["units1"])
    units2 = int(dense_cfg["units2"])

    with mlflow.start_run():
        # Tags for MLflow
        mlflow.set_tag("model", "cnn_custom_dense")
        mlflow.set_tag("dev", "raoul")

        # Log hyperparameters
        mlflow.log_param("filters", filters)
        mlflow.log_param("units1", units1)
        mlflow.log_param("units2", units2)
        mlflow.log_param("batchsize", batchsize)

        # Create model on the correct device
        model = CNN(filters=filters, units1=units1, units2=units2).to(device)

        # Trainer (reusing your existing settings / loaders)
        trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=loss_fn,
            optimizer=optimizer_cls,
            traindataloader=trainstreamer,
            validdataloader=validstreamer,
            scheduler=optim.lr_scheduler.ReduceLROnPlateau,
            device=device,
        )

        trainer.loop()

        # Save the trained model
        tag = datetime.now().strftime("%Y%m%d-%H%M")
        modelpath = modeldir / f"{tag}_model.pt"
        torch.save(model, modelpath)

        # Log as MLflow artifact
        mlflow.log_artifact(
            local_path=str(modelpath), artifact_path="pytorch_models"
        )

        # Hyperopt minimizes "loss"
        return {"loss": trainer.test_loss, "status": STATUS_OK}


# ---------------------------------------------------------------------
# Hyperparameter search space & optimization
# ---------------------------------------------------------------------
search_space = {
    # Number of convolution filters
    "filters": scope.int(hp.quniform("filters", 16, 128, 16)),
    # Discrete choices for (units1, units2)
    "dense_config": hp.choice(
        "dense_config",
        [
            {"units1": 64, "units2": 64},
            {"units1": 128, "units2": 64},
            {"units1": 128, "units2": 128},
        ],
    ),
}

trials = Trials()
best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=48,
    trials=trials,
)
print("best_result:", best_result)
