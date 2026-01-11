from pathlib import Path
import numpy as np
import torch
import mlflow
from datetime import datetime
from typing import List
from torch.nn.utils.rnn import pad_sequence
from mltrainer import rnn_models, Trainer
from torch import optim

from mads_datasets import datatools
import mltrainer
mltrainer.__version__

from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer.preprocessors import PaddedPreprocessor
preprocessor = PaddedPreprocessor()

gesturesdatasetfactory = DatasetFactoryProvider.create_factory(DatasetType.GESTURES)
streamers = gesturesdatasetfactory.create_datastreamer(batchsize=32, preprocessor=preprocessor)
train = streamers["train"]
valid = streamers["valid"]

print("len(train) | len(valid):",len(train), len(valid))  # 81 20
                                                          # train is 81 x 32 batches = 2529 gestures
                                                          # valid is 20 x 32 batches = 640 gestures 

trainstreamer = train.stream()
validstreamer = valid.stream()
x, y = next(iter(trainstreamer))
print("x.shape | y:",x.shape, y)  # x.shape | y: torch.Size([32, 34, 3]) tensor([ 3,  0,  5,  3,  0,  6, 14,  7, 12, 17, 12,  6,  3, 14, 16, 15, 12, 18,
                                  #                                              10,  0,  6, 17,  6, 19,  6, 14,  3,  0, 11, 15,  7, 10])

# Can you make sense of the shape? AvT: Train -> 32 batches, each with 34 time steps, each time step has 3 features (x,y,z accelerometer)
#                                       Test  -> 32 values of the actual gesture, with a value between 0 and 19
# What does it mean that the shapes are sometimes (32, 27, 3), but a second time might look like (32, 30, 3)? In other words, the second (or first, if you insist on starting at 0) dimension changes.
# Why is that? AvT: Some gestures are shorter and quicker, meaning not all 34 timesteps have a values. Example: complete gesture was done after 27 timesteps?
# How does the model handle this? Do you think this is already padded, or still has to be padded? AvT: For sure if variable length, it still has to be padded.
# Likely the PaddedPreprocessor arranges all padding.

# Excercises
# Lets test a basemodel, and try to improve upon that.
# Fill the gestures.gin file with relevant settings for `input_size`, `hidden_size`, `num_layers` and `horizon` (which, in our case, will be the number of classes...)
# As a rule of thumbs: start lower than you expect to need!

from mltrainer import TrainerSettings, ReportTypes
from mltrainer.metrics import Accuracy

accuracy = Accuracy()

model = rnn_models.BaseRNN(
    input_size=3,
    hidden_size=64,
    num_layers=1,
    horizon=20,
)

# ---
# Explaination
# input_size=3 -> x,y,z accelaration
# hidden_size=64
# num_layers=1
# horizon=20 -> 20 gesture classes (0-19)
#
# Why is it called horizon? (This confuses everyone!)
# For sequence classification (like gestures, emotion recognition, etc.), horizon is used to classify into N classes (here N=20) 
#
# How the model works:
# - Takes input: (batch_size, 34, 3)
# - Runs it through 1 layer of GRU/LSTM with hidden size 64
# - Takes the last valid hidden state (after 34 steps, ignoring padding)
# - Puts it through a fully connected layer: 64 -> 20
# - Outputs logits of shape (batch_size, 20)
# - Apply softmax → probability for each of the 20 gestures
# ---

# Test the model. What is the output shape you need? Remember, we are doing classification!
yhat = model(x)
print("yhat.shape:", yhat.shape)  # torch.Size([32, 20])

# Test the accuracy
accuracy(y, yhat)

# What do you think of the accuracy? What would you expect from blind guessing?
# Check shape of `y` and `yhat`
print("yhat.shape | y.shape:", yhat.shape, y.shape)  # yhat.shape | y.shape: torch.Size([32, 20]) torch.Size([32])

# And look at the output of yhat
print("yhat[0]:", yhat[0])  # yhat[0]: tensor([-0.1670, -0.0849,  0.0772, -0.0418, -0.0598,  0.0649,  0.0228,  0.0680,
                            #                  -0.0522,  0.2553,  0.1395, -0.1375,  0.0914, -0.0599,  0.1184,  0.1090,
                            #                   0.0410, -0.1188,  0.1572, -0.0254], grad_fn=<SelectBackward0>)

# Does this make sense to you? If you are unclear, go back to the classification problem with the MNIST, where we had 10 classes.
# AvT: As stated above, y has shape [32,1]: 32 values of the actual gesture, with a value between 0 and 19.
#      yhat has shape [32,20]: for each of the 32 samples, 20 values representing the scores for each class (gesture 0-19). 
# We have a classification problem, so we need Cross Entropy Loss.
# Remember, [this has a softmax built in](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(yhat, y)
print("loss:", loss)

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

# On my mac, at least for the BaseRNN model, mps does not speed up training
# Probably because the overhead of copying the data to the GPU is too high
# So i override the device to cpu
device = "cpu"
# However, it might speed up training for larger models, with more parameters

# Set up the settings for the trainer and the different types of logging you want
settings = TrainerSettings(
    epochs=3, # increase this to about 100 for training
    metrics=[accuracy],
    logdir=Path("gestures"),
    train_steps=len(train),
    valid_steps=len(valid),
    reporttypes=[ReportTypes.TOML, ReportTypes.TENSORBOARD, ReportTypes.MLFLOW],
    scheduler_kwargs={"factor": 0.5, "patience": 5},
    earlystop_kwargs = {
        "save": False, # save every best model, and restore the best one
        "verbose": True,
        "patience": 5, # number of epochs with no improvement after which training will be stopped
        "delta": 0.0, # minimum change to be considered an improvement
    }
)
print("settings:")
print("=========")
print(settings)

# settings:
# =========
# epochs: 3
# metrics: [Accuracy]
# logdir: gestures
# train_steps: 81
# valid_steps: 20
# reporttypes: [<ReportTypes.TOML: 'TOML'>, <ReportTypes.TENSORBOARD: 'TENSORBOARD'>, <ReportTypes.MLFLOW: 'MLFLOW'>]
# optimizer_kwargs: {'lr': 0.001, 'weight_decay': 1e-05}
# scheduler_kwargs: {'factor': 0.5, 'patience': 5}
# earlystop_kwargs: {'save': False, 'verbose': True, 'patience': 5, 'delta': 0.0}

# ---
# Explaining scheduler_kwargs
#
# What it is: Keyword arguments for the learning-rate scheduler ReduceLROnPlateau.
# What ReduceLROnPlateau does: It reduces the learning rate when a monitored metric stops improving (usually the validation loss).
# The specific arguments:
# - factor=0.5 -> When the scheduler decides to reduce the LR, it multiplies the current LR by 0.5. Example: if LR = 0.001, it becomes 0.0005.
# - patience=5 -> The scheduler waits for 5 epochs with no improvement in the monitored metric before reducing the LR.
# So in words: “If the validation metric doesn’t get better for 5 epochs, cut the learning rate in half.”
# ---

import torch.nn as nn
import torch
from torch import Tensor
from dataclasses import dataclass
from pathlib import Path
import mlflow

@dataclass
class ModelConfig:
    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int
    dropout: float = 0.0

class GRUmodel(nn.Module):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.config = config
        self.rnn = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            batch_first=True,
            num_layers=config.num_layers,
        )
        self.linear = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat

config = ModelConfig(
    input_size=3,
    hidden_size=64,
    num_layers=1,
    output_size=20,
    dropout=0.0,
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("gestures")
modeldir = Path("gestures").resolve()
if not modeldir.exists():
    modeldir.mkdir(parents=True)

with mlflow.start_run():
    mlflow.set_tag("model", "nnGRU")
    mlflow.set_tag("dev", "in3_h64_nl1_out20_drop00")
    config = ModelConfig(
        input_size=3,
        hidden_size=64,
        num_layers=1,
        output_size=20,
        dropout=0.1,
    )

    model = GRUmodel(
        config=config,
    )

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

    if not settings.earlystop_kwargs["save"]:
        tag = datetime.now().strftime("%Y%m%d-%H%M-")
        modelpath = modeldir / (tag + "model.pt")
        torch.save(model, modelpath)

# Try to update the code above by changing the hyperparameters.
# To discern between the changes, also modify the tag mlflow.set_tag("model", "new-tag-here") where you add a new tag of your choice. This way you can keep the models apart.

# trainer.loop() # if you want to pick up training, loop will continue from the last epoch

mlflow.end_run()
print("mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
print("Then open: http://127.0.0.1:5000")

# Excercises:
# - try to improve the RNN model
# - test different things. What works? What does not?
# - experiment with either GRU or LSTM layers, create your own models. Have a look at `mltrainer.rnn_models` for inspiration. 
# - experiment with adding Conv1D layers. Think about the necessary input-output dimensions of your tensors before and after each layer.
# 
# You should be able to get above 90% accuracy with the dataset.
# Create a report of 1 a4 about your experiments.