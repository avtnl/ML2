from pathlib import Path
import torch
import torch.nn as nn
from loguru import logger
import warnings
warnings.simplefilter("ignore", UserWarning)

from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer.preprocessors import BasePreprocessor

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

# ==================== DEVICE (GPU first, then CPU) ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
logger.info(f"Using device: {device}")

# Define model
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
            nn.Linear(units2, 10)
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

# Optional: old CNN, not used below (kept for reference)
# model = CNN(filters=32, units1=128, units2=64).to(device)

from mltrainer.imagemodels import CNNConfig, CNNblocks

config = CNNConfig(
    matrixshape=(28, 28),  # every image is 28x28
    batchsize=batchsize,
    input_channels=1,      # we have black and white images, so only one channel
    hidden=32,             # number of filters
    kernel_size=3,         # kernel size of the convolution
    maxpool=3,             # kernel size of the maxpool
    num_layers=4,          # we will stack 4 Convolutional blocks, each with two Conv2d layers
    num_classes=10,
)

# IMPORTANT: move CNNblocks model to the chosen device
model = CNNblocks(config).to(device)
model.config

from torchinfo import summary
summary(model, input_size=(32, 1, 28, 28))

import torch.optim as optim
from mltrainer import metrics
optimizer = optim.Adam
loss_fn = torch.nn.CrossEntropyLoss()
accuracy = metrics.Accuracy()

# Move test batch to the same device and run a quick sanity check
x = x.to(device)
y = y.to(device)
yhat = model(x)
accuracy(y, yhat)

from mltrainer import metrics, Trainer, TrainerSettings, ReportTypes
settings = TrainerSettings(
    epochs=3,
    metrics=[accuracy],
    logdir="demo",
    train_steps=100,
    valid_steps=100,
    reporttypes=[ReportTypes.TOML],
)

trainer = Trainer(
    model=model,
    settings=settings,
    loss_fn=loss_fn,
    optimizer=optimizer,
    traindataloader=trainstreamer,
    validdataloader=validstreamer,
    scheduler=optim.lr_scheduler.ReduceLROnPlateau,
    device=device,
)
trainer.loop()
