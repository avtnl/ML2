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

import torch
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using MPS")
elif torch.cuda.is_available():
    device = "cuda:0"
    print("using cuda")
else:
    device = "cpu"
    print("using cpu")

from torch import nn
print(f"Using {device} device")

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

    def _conv_test(self, input_size = (32, 1, 28, 28)):
        x = torch.ones(input_size)
        x = self.convolutions(x)
        return x.shape[-2:]

    def forward(self, x):
        x = self.convolutions(x)
        x = self.agg(x)
        logits = self.dense(x)
        return logits

model = CNN(filters=32, units1=128, units2=64).to("cpu")

from mltrainer.imagemodels import CNNConfig, CNNblocks

config = CNNConfig(
    matrixshape = (28, 28), # every image is 28x28
    batchsize = batchsize,
    input_channels = 1, # we have black and white images, so only one channel
    hidden = 32, # number of filters
    kernel_size = 3, # kernel size of the convolution
    maxpool = 3, # kernel size of the maxpool
    num_layers = 4, # we will stack 4 Convolutional blocks, each with two Conv2d layers
    num_classes = 10,
)

model = CNNblocks(config)
model.config

from torchinfo import summary
summary(model, input_size=(32, 1, 28, 28))

import torch.optim as optim
from mltrainer import metrics
optimizer = optim.Adam
loss_fn = torch.nn.CrossEntropyLoss()
accuracy = metrics.Accuracy()

yhat = model(x.to("cpu"))
accuracy(y.to("cpu"), yhat)

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

# DatasetType.FLOWERS
# DatasetType.IMDB
# DatasetType.GESTURES
# DatasetType.FASHION
# DatasetType.SUNSPOTS
# DatasetType.IRIS
# DatasetType.PENGUINS
# DatasetType.FAVORITA
# DatasetType.SECURE
# 2025-11-28 13:33:22.720 | INFO     | mads_datasets.base:download_data:121 - Folder already exists at C:\Users\avtnl\.cache\mads_datasets\fashionmnist
# 2025-11-28 13:33:22.720 | INFO     | mads_datasets.base:download_data:124 - File already exists at C:\Users\avtnl\.cache\mads_datasets\fashionmnist\fashionmnist.pt
# using cuda
# Using cuda:0 device
# 2025-11-28 13:33:25.497 | INFO     | __main__:__init__:63 - Aggregating activationmap with size torch.Size([2, 2])
# Calculated matrix size: 9
# Caluclated flatten size: 288
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# CNNblocks                                [32, 10]                  --
# ├─ModuleList: 1-1                        --                        --
# │    └─ConvBlock: 2-1                    [32, 32, 28, 28]          --
# │    │    └─Sequential: 3-1              [32, 32, 28, 28]          9,568
# │    └─ConvBlock: 2-2                    [32, 32, 28, 28]          --
# │    │    └─Sequential: 3-2              [32, 32, 28, 28]          18,496
# │    └─ReLU: 2-3                         [32, 32, 28, 28]          --
# │    └─MaxPool2d: 2-4                    [32, 32, 9, 9]            --
# │    └─ConvBlock: 2-5                    [32, 32, 9, 9]            --
# │    │    └─Sequential: 3-3              [32, 32, 9, 9]            18,496
# │    └─ReLU: 2-6                         [32, 32, 9, 9]            --
# │    └─ConvBlock: 2-7                    [32, 32, 9, 9]            --
# │    │    └─Sequential: 3-4              [32, 32, 9, 9]            18,496
# │    └─ReLU: 2-8                         [32, 32, 9, 9]            --
# │    └─MaxPool2d: 2-9                    [32, 32, 3, 3]            --
# │    └─ConvBlock: 2-10                   [32, 32, 3, 3]            --
# │    │    └─Sequential: 3-5              [32, 32, 3, 3]            18,496
# │    └─ReLU: 2-11                        [32, 32, 3, 3]            --
# ├─Sequential: 1-2                        [32, 10]                  --
# │    └─Flatten: 2-12                     [32, 288]                 --
# │    └─Linear: 2-13                      [32, 32]                  9,248
# │    └─ReLU: 2-14                        [32, 32]                  --
# │    └─Linear: 2-15                      [32, 10]                  330
# ==========================================================================================
# Total params: 93,130
# Trainable params: 93,130
# Non-trainable params: 0
# Total mult-adds (Units.MEGABYTES): 805.59
# ==========================================================================================
# Input size (MB): 0.10
# Forward/backward pass size (MB): 28.50
# Params size (MB): 0.37
# Estimated Total Size (MB): 28.98
# ==========================================================================================