import torch
import torch.nn as nn
from loguru import logger
from torchinfo import summary
import time
import copy
from pathlib import Path
import warnings
warnings.simplefilter("ignore", UserWarning)


import mltrainer
from mltrainer.preprocessors import BasePreprocessor
import mads_datasets
from mads_datasets import DatasetFactoryProvider, DatasetType
from tomlserializer import TOMLSerializer

fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)

print("version mads_datasets: ", mads_datasets.__version__)

datasets = fashionfactory.create_dataset()

# we can either use pytorches DataLoader
from torch.utils.data import DataLoader
trainloader = DataLoader(datasets["train"], batch_size=64, shuffle=True)
testloader = DataLoader(datasets["valid"], batch_size=64, shuffle=True)

start = time.perf_counter()
X, y = next(iter(trainloader))
end = time.perf_counter()

print(f"Execution time dataloader: {end - start:.6f} seconds")

# or the BaseDatastreamer from the datasetfactory. Check out which one is faster
streamers = fashionfactory.create_datastreamer(batchsize=64, preprocessor=BasePreprocessor())
train = streamers["train"]
valid = streamers["valid"]
trainstreamer = train.stream()
validstreamer = valid.stream()
X, y = next(iter(trainstreamer))
X.shape

start = time.perf_counter()
X, y = next(iter(trainstreamer))
end = time.perf_counter()

print(f"Execution time datastreamer: {end - start:.6f} seconds")

len(train), len(valid)
print("length dataset: "+str(len(train))+","+str(len(valid)))

X, y = next(iter(trainstreamer))

#X.shape, y.shape
print("torch size: "+str(X.shape)+","+str(y.shape))

#type(X[0])
print("type of X: "+str(type(X[0])))

img = X[1]
img.shape
print("image size: "+str(img.shape))

#Print 1st image
import matplotlib.pyplot as plt
plt.imshow(img.squeeze(), cmap="gray")

# Create a model
import torch

# FORCED CORRECT WAY – works on your G14 every single time
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA IS WORKING! Using GPU → {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
else:
    device = torch.device("cpu")
    print("CUDA not available → falling back to CPU")

print(f"Final device used everywhere: {device}")

logger.info(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.units_1 = 512
        self.units_2 = 256
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, self.units_1),
            nn.ReLU(),
            nn.Linear(self.units_1, self.units_2),
            nn.ReLU(),
            nn.Linear(self.units_2, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to("cpu")
from torchinfo import summary
summary(model, input_size=(1, 28, 28))

#Optimizer
import torch.optim as optim
loss_fn = torch.nn.CrossEntropyLoss()

X, y = next(iter(trainstreamer))
model.to(device)
next(model.parameters()).is_cuda

yhat = model(X.to(device)) # make a prediction

loss_fn(yhat, y.to(device)) # calculate the loss

log_dir=Path("log_AvT").absolute()
log_dir

from mltrainer import metrics
accuracy = metrics.Accuracy()

from mltrainer import TrainerSettings, ReportTypes, Trainer

settings = TrainerSettings(
    epochs=3,
    metrics=[accuracy],
    logdir=log_dir,
    train_steps=len(train),
    valid_steps=len(valid),
    reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.TOML],
)
settings
logger.info(f"TrainerSettings:\n{settings}")

#Train loop
trainer = Trainer(
    model=model,
    settings=settings,
    loss_fn=loss_fn,
    optimizer=optim.Adam,
    traindataloader=trainstreamer,
    validdataloader=validstreamer,
    scheduler=optim.lr_scheduler.ReduceLROnPlateau,
    device=device,
    )

trainer.loop()

#Save manually
modeldir = Path("AvT/saved_model")
modelpath = modeldir / "trained_model"
if not modeldir.exists():
    modeldir.mkdir(parents=True)
    torch.save(model, modelpath)
    logger.info(f"Model saved to {modelpath}")

print("early stop path:"+str(trainer.early_stopping.path))

#load the model
# note that I would expect the loaded model to run on mps, but that doesnt work as expected
if device == "mps":
    device = "cpu"
print(f"using {device}")
loaded_model = torch.load(modelpath, map_location=device, weights_only=False)

# show that all parameters are on the same device, and use acceleration if available
for param in loaded_model.parameters():
    print(param.device)

# Get a batch $X$, $y$ and make a prediction yhat
X, y = next(iter(testloader))
yhat = loaded_model(X.to(device))
loss_fn(yhat, y.to(device))
acc = (yhat.argmax(dim=1) == y.to(device)).sum() / len(y.to(device))
print("accuracy:"+str(acc.item() * 100))

