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
from mltrainer.imagemodels import CNNConfig, CNNblocks
from torchinfo import summary

import mlflow
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
warnings.simplefilter("ignore", UserWarning)

for dataset in DatasetType:
    print(dataset)

fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
batchsize = 64
preprocessor = BasePreprocessor()
streamers = fashionfactory.create_datastreamer(batchsize=batchsize, preprocessor=preprocessor)
train = streamers["train"]
valid = streamers["valid"]
trainstreamer = train.stream()
validstreamer = valid.stream()

x, y = next(iter(trainstreamer))
x.shape, y.shape

# ---------------------------------------------------------------------
# Device (GPU first, then CPU)
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
logger.info(f"Using device: {device}")

# ---------------------------------------------------------------------
# (Optional) Simple CNN definition kept for reference
# ---------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, filters, units1, units2, input_size=(32, 1, 28, 28)):
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
        )

        activation_map_size = self._conv_test(input_size)
        logger.info(f"Aggregating activationmap with size {activation_map_size}")
        self.agg = nn.AvgPool2d(activation_map_size)

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters, units1),
            nn.ReLU(),
            nn.Linear(units1, units2),
            nn.ReLU(),
            nn.Linear(units2, 10),
        )

    def _conv_test(self, input_size=(32, 1, 28, 28)):
        x = torch.ones(input_size)
        x = self.convolutions(x)
        return x.shape[-2:]

    def forward(self, x):
        x = self.convolutions(x)
        x = self.agg(x)
        logits = self.dense(x)
        return logits

# ---------------------------------------------------------------------
# Default CNNblocks config (for summary / sanity check)
# ---------------------------------------------------------------------
default_config = CNNConfig(
    matrixshape=(28, 28),  # every image is 28x28
    batchsize=batchsize,
    input_channels=1,      # black & white → 1 channel
    hidden=32,             # number of filters
    kernel_size=3,         # convolution kernel size
    maxpool=3,             # maxpool kernel size
    num_layers=4,          # number of conv blocks
    num_classes=10,
)

default_model = CNNblocks(default_config).to(device)
default_model.config

summary(default_model, input_size=(32, 1, 28, 28))

# Quick sanity check on one batch
accuracy = metrics.Accuracy()
x = x.to(device)
y = y.to(device)
with torch.no_grad():
    yhat = default_model(x)
accuracy(y, yhat)

# ---------------------------------------------------------------------
# MLflow + Hyperopt setup
# ---------------------------------------------------------------------
modeldir = Path("models").resolve()
if not modeldir.exists():
    modeldir.mkdir()
    print(f"Created {modeldir}")

settings = TrainerSettings(
    epochs=10,
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
      - tunes only the number of filters in the conv layers (hidden)
      - keeps kernel_size and num_layers fixed to a safe configuration
    """
    # Make sure Hyperopt's value is an int
    filters = int(params["filters"])

    with mlflow.start_run():
        # Tags for MLflow
        mlflow.set_tag("model", "cnnblocks")
        mlflow.set_tag("dev", "raoul")

        # Log hyperparameters
        mlflow.log_param("filters", filters)
        mlflow.log_param("kernel_size", 3)   # fixed, safe value
        mlflow.log_param("num_layers", 4)    # fixed, safe value
        mlflow.log_param("batchsize", batchsize)

        # Build CNNConfig with fixed geometry + tuned filters
        config = CNNConfig(
            matrixshape=(28, 28),
            batchsize=batchsize,
            input_channels=1,
            hidden=filters,   # this is what Hyperopt is tuning
            kernel_size=3,    # fixed
            maxpool=3,        # as in mlt04b
            num_layers=4,     # fixed
            num_classes=10,
        )

        # Create model on the correct device
        model = CNNblocks(config).to(device)

        # Trainer (reusing your existing settings / loaders)
        trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=loss_fn,
            optimizer=optimizer_cls,  # or optimizer, depending on how you named it
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
        mlflow.log_artifact(local_path=str(modelpath), artifact_path="pytorch_models")

        # Hyperopt minimizes "loss"
        return {"loss": trainer.test_loss, "status": STATUS_OK}

# ---------------------------------------------------------------------
# Hyperparameter search space & optimization
# ---------------------------------------------------------------------
search_space = {
    # Number of convolution filters (CNNConfig.hidden)
    "filters": scope.int(hp.quniform("filters", 16, 128, 8)),
}

trials = Trials()
best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=32,
    trials=trials,
)
print("best_result:", best_result)

