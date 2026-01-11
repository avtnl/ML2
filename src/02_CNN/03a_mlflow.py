import torch
import torch.nn as nn
from torchinfo import summary
from loguru import logger
from pathlib import Path
import warnings
warnings.simplefilter("ignore", UserWarning)
import mltrainer
mltrainer.__version__
import copy

print("version mltrainer: ", mltrainer.__version__)

#Load data
from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer.preprocessors import BasePreprocessor
preprocessor = BasePreprocessor()

fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
streamers = fashionfactory.create_datastreamer(batchsize=64, preprocessor=preprocessor)
# flowersfactory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
# streamers = flowersfactory.create_datastreamer(batchsize=32, preprocessor=preprocessor)
train = streamers["train"]
valid = streamers["valid"]

print("len(train), len(valid): "+str(len(train))+","+str(len(valid)))  # len(train), len(valid): 937,156

# Obtain 1 item
trainstreamer = train.stream()
validstreamer = valid.stream()
x, y = next(iter(trainstreamer))
x.shape, y.shape
print("x.shape: "+str(x.shape))  # x.shape: torch.Size([64, 1, 28, 28])
print("y.shape: "+str(y.shape))  # y.shape: torch.Size([64])

# Let's pull this 1 item through a Conv2d layer:
in_channels = x.shape[1]
print("in_channels: "+str(in_channels))  # in_channels: 1

conv = nn.Conv2d(
    in_channels=in_channels,
    out_channels=64,
    kernel_size=3,
    padding=(1,1))
out = conv(x)
out.shape
print("shape1: "+str(out.shape))  # shape: torch.Size([64, 64, 28, 28])

conv = nn.Conv2d(
    in_channels=in_channels,
    out_channels=64,
    kernel_size=3,
    padding=(0,0))
out = conv(x)
out.shape
print("shape2: "+str(out.shape))  # shape: torch.Size([64, 64, 26, 26])

conv = nn.Conv2d(
    in_channels=in_channels,
    out_channels=64,
    kernel_size=3,
    padding=(0,0),
    stride=2)
out = conv(x)
out.shape
print("shape3: "+str(out.shape))  # shape: torch.Size([64, 64, 13, 13])

convolutions = nn.Sequential(
    nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
)
out = convolutions(x)
out.shape
print("shape4: "+str(out.shape))  # shape: torch.Size([64, 32, 14, 14])

### NEW ###

convolutions = nn.Sequential(
    nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),  # shape: torch.Size([64, 32, 28,28])
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),  # shape: torch.Size([64, 32, 14, 14])
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),  # shape: torch.Size([64, 32, 12, 12])
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),  # shape: torch.Size([64, 32, 6, 6])
)
out = convolutions(x)
out.shape
print("shape5: "+str(out.shape))  # shape: torch.Size([64, 32, 6, 6])

### NEW ###

convolutions = nn.Sequential(
    nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),  # shape: torch.Size([64, 32, 28,28])
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),   # shape: torch.Size([64, 32, 14, 14])
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),  # shape: torch.Size([64, 32, 12, 12])
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),  # shape: torch.Size([64, 32, 6, 6])
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),  # shape: torch.Size([64, 32, 4, 4])
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),  # shape: torch.Size([64, 32, 2, 2])
)
out = convolutions(x)
out.shape
print("shape6: "+str(out.shape))  # shape: torch.Size([64, 32, 2, 2])

### NEW ###

convolutions = nn.Sequential(
    nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
)
out = convolutions(x)
input_nn = nn.Flatten()(out)
input_nn.shape
print("shape7: "+str(input_nn.shape))  # shape: torch.Size([64, 128])

### NEW ###

convolutions = nn.Sequential(
    nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
    nn.ReLU(),
    nn.AvgPool2d((2,2)),
)
out = convolutions(x)
out.shape
print("shape8: "+str(out.shape))  # shape: torch.Size([64, 32, 2, 2])

input_nn = nn.Flatten()(out)
input_nn.shape
print("shape9: "+str(input_nn.shape))  # shape: torch.Size([64, 128])

### NEW ###

# Define model
class CNN(nn.Module):
    def __init__(self, filters: int, units1: int, units2: int, input_size: tuple):
        super().__init__()
        self.in_channels = input_size[1] # (batch x channels x height x width), so we need the second element
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

        activation_map_size = self._conv_test(self.input_size)
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

    def _conv_test(self, input_size):
        x = torch.ones(input_size, dtype=torch.float32)
        x = self.convolutions(x)
        return x.shape[-2:]

    def forward(self, x):
        x = self.convolutions(x)
        x = self.agg(x)
        logits = self.dense(x)
        return logits

model = CNN(filters=128, units1=128, units2=64, input_size=(32, 3, 224, 224))
summary(model, input_size=(32, 3, 224, 224), device="cpu")
print("summary: "+str(summary))

# 2025-11-27 17:54:50.646 | INFO     | __main__:__init__:163 - Aggregating activationmap with size torch.Size([26, 26])
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# CNN                                      [32, 10]                  --
# ├─Sequential: 1-1                        [32, 128, 26, 26]         --
# │    └─Conv2d: 2-1                       [32, 128, 224, 224]       3,584
# │    └─ReLU: 2-2                         [32, 128, 224, 224]       --
# │    └─MaxPool2d: 2-3                    [32, 128, 112, 112]       --
# │    └─Conv2d: 2-4                       [32, 128, 110, 110]       147,584
# │    └─ReLU: 2-5                         [32, 128, 110, 110]       --
# │    └─MaxPool2d: 2-6                    [32, 128, 55, 55]         --
# │    └─Conv2d: 2-7                       [32, 128, 53, 53]         147,584
# │    └─ReLU: 2-8                         [32, 128, 53, 53]         --
# │    └─MaxPool2d: 2-9                    [32, 128, 26, 26]         --
# ├─AvgPool2d: 1-2                         [32, 128, 1, 1]           --
# ├─Sequential: 1-3                        [32, 10]                  --
# │    └─Flatten: 2-10                     [32, 128]                 --
# │    └─Linear: 2-11                      [32, 128]                 16,512
# │    └─ReLU: 2-12                        [32, 128]                 --
# │    └─Linear: 2-13                      [32, 64]                  8,256
# │    └─ReLU: 2-14                        [32, 64]                  --
# │    └─Linear: 2-15                      [32, 10]                  650
# ==========================================================================================
# Total params: 324,170
# Trainable params: 324,170
# Non-trainable params: 0
# Total mult-adds (Units.GIGABYTES): 76.17
# ==========================================================================================
# Input size (MB): 19.27
# Forward/backward pass size (MB): 2132.76
# Params size (MB): 1.30
# Estimated Total Size (MB): 2153.32
# ==========================================================================================
# summary: <function summary at 0x0000020FCFF0BC40>



model = CNN(filters=128, units1=128, units2=64, input_size=(32, 1, 28, 28))
summary(model, input_size=(32, 1, 28, 28), device="cpu")
print("summary: "+str(summary))

# 2025-11-27 17:54:51.505 | INFO     | __main__:__init__:163 - Aggregating activationmap with size torch.Size([2, 2])
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# CNN                                      [32, 10]                  --
# ├─Sequential: 1-1                        [32, 128, 2, 2]           --
# │    └─Conv2d: 2-1                       [32, 128, 28, 28]         1,280
# │    └─ReLU: 2-2                         [32, 128, 28, 28]         --
# │    └─MaxPool2d: 2-3                    [32, 128, 14, 14]         --
# │    └─Conv2d: 2-4                       [32, 128, 12, 12]         147,584
# │    └─ReLU: 2-5                         [32, 128, 12, 12]         --
# │    └─MaxPool2d: 2-6                    [32, 128, 6, 6]           --
# │    └─Conv2d: 2-7                       [32, 128, 4, 4]           147,584
# │    └─ReLU: 2-8                         [32, 128, 4, 4]           --
# │    └─MaxPool2d: 2-9                    [32, 128, 2, 2]           --
# ├─AvgPool2d: 1-2                         [32, 128, 1, 1]           --
# ├─Sequential: 1-3                        [32, 10]                  --
# │    └─Flatten: 2-10                     [32, 128]                 --
# │    └─Linear: 2-11                      [32, 128]                 16,512
# │    └─ReLU: 2-12                        [32, 128]                 --
# │    └─Linear: 2-13                      [32, 64]                  8,256
# │    └─ReLU: 2-14                        [32, 64]                  --
# │    └─Linear: 2-15                      [32, 10]                  650
# ==========================================================================================
# Total params: 321,866
# Trainable params: 321,866
# Non-trainable params: 0
# Total mult-adds (Units.MEGABYTES): 788.56
# ==========================================================================================
# Input size (MB): 0.10
# Forward/backward pass size (MB): 30.98
# Params size (MB): 1.29
# Estimated Total Size (MB): 32.37
# ==========================================================================================
# summary: <function summary at 0x0000020FCFF0BC40>

import torch.optim as optim
from mltrainer import metrics
from mltrainer import Trainer, TrainerSettings, ReportTypes

optimizer = optim.Adam
loss_fn = torch.nn.CrossEntropyLoss()
accuracy = metrics.Accuracy()

print("version mltrainer: ", mltrainer.__version__)

model = CNN(filters=128, units1=128, units2=64, input_size=(32, 1, 28, 28))

yhat = model(x)
accuracy(y, yhat)

log_dir = Path("modellog").resolve()
if not log_dir.exists():
    log_dir.mkdir(parents=True)

#TrainingSettings
settings = TrainerSettings(
    epochs=3,
    metrics=[accuracy],
    logdir=log_dir,
    train_steps=len(train),
    valid_steps=len(valid),
    reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.TOML],
)
settings

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using MPS")
elif torch.cuda.is_available():
    device = "cuda:0"
    print("using cuda")
else:
    device = "cpu"
    print("using cpu")

#Training
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

# version mltrainer:  0.2.5
# 2025-11-28 13:22:49.456 | INFO     | __main__:__init__:162 - Aggregating activationmap with size torch.Size([2, 2])
# using cuda
# 2025-11-28 13:22:52.413 | INFO     | mltrainer.trainer:dir_add_timestamp:24 - Logging to C:\Users\avtnl\Documents\HU\ML2\modellog\20251128-132252
# 2025-11-28 13:22:55.056 | INFO     | mltrainer.trainer:__init__:68 - Found earlystop_kwargs in settings.Set to None if you dont want earlystopping.
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 937/937 [00:08<00:00, 105.09it/s]
# 2025-11-28 13:23:04.851 | INFO     | mltrainer.trainer:report:209 - Epoch 0 train 0.7221 test 0.4934 metric ['0.8187']██████████████████████████████████████████████████████▍| 934/937 [00:08<00:00, 122.94it/s] 
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 937/937 [00:07<00:00, 120.92it/s]
# 2025-11-28 13:23:13.217 | INFO     | mltrainer.trainer:report:209 - Epoch 1 train 0.4235 test 0.3762 metric ['0.8637']█████████████████████████████████████████████████████▌ | 929/937 [00:07<00:00, 120.38it/s] 
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 937/937 [00:08<00:00, 116.43it/s]
# 2025-11-28 13:23:21.872 | INFO     | mltrainer.trainer:report:209 - Epoch 2 train 0.3443 test 0.3291 metric ['0.8808']█████████████████████████████████████████████████████▍ | 928/937 [00:07<00:00, 116.80it/s] 
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:26<00:00,  8.86s/it]
# 