from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore", UserWarning)
#%matplotlib inline
plt.style.use('dark_background')
import mltrainer
mltrainer.__version__

# Loading data
from mads_datasets import DatasetFactoryProvider, DatasetType
sunspotfactory = DatasetFactoryProvider.create_factory(DatasetType.SUNSPOTS)
sunspotfactory.download_data()

# Show raw data 
import numpy as np
f = sunspotfactory.filepath
print("filepath")
print(f.exists(), f)

spots = np.genfromtxt(f, usecols=(2,3))
print("spots")
print(spots)

# This is data on sunspots, since 1749, in a pandas dataframe. Let us cast this to a `float32` Torch Tensor.
# We first need to get the numpy data out of the pandas, and the cast that to a float32 Tensor.

import torch
tensordata = torch.from_numpy(spots[:, 1]).type(torch.float32)
print("tensordata: "+str(tensordata))  # Only contains the average number of sunspots per month, nothing regarding time!
print(tensordata.shape)       # -> torch.Size([3000])
print(tensordata.dtype)       # -> torch.float32
print(tensordata[:5])         # first 5 sunspot values


# ---
# Explanation
# spots[:, 1] – take the second column
# spots is a NumPy array, probably of shape (N, 2).
# spots[:, 1] means:
# : → all rows
# 1 → column with index 1 (the second column)
# Result: a 1D NumPy array of length N, containing just that column:
# ---

plt.plot(tensordata)
plt.show()

# Train test split
split = 2500
train = tensordata[:split]
print("train")
print(train.shape)         # torch.Size([3323])
print(train.dtype)         # torch.float32
print(train[:5])           # tensor([ 96.7000, 104.3000, 116.7000,  92.8000, 141.7000])-

norm = max(train)
print("norm: "+str(norm))  # tensor(398.2000)
test = tensordata[split:]
print("test")
print(test.shape)          # torch.Size([2500])
print(test.dtype)          # torch.float32
print(test[:5])            # tensor([233.0000, 284.3000, 265.1000, 223.7000, 334.0000])

train = train / norm
print("train/ norm") 
print(train.shape)         # torch.Size([2500])
print(train.dtype)         # torch.float32
print(train[:5])           # tensor([0.2428, 0.2619, 0.2931, 0.2330, 0.3559])

test = test/ norm
print("test/ norm")
print(test.shape)          # torch.Size([823])
print(test.dtype)          # torch.float32
print(test[:5])            # tensor([0.5851, 0.7140, 0.6657, 0.5618, 0.8388])
#print(test) 

# What would go wrong if you normalize on the max of the testset?

# Reduce steps back to months, starting in 1749
idx0 = torch.arange(0, len(train)) / 12 + 1749
idx1 = torch.arange(0, len(test)) / 12 + idx0.max()
fig, axs = plt.subplots(2, sharex=True)
axs[0].plot(idx0, train)
axs[1].plot(idx1,test)
plt.show()

# Windowing
from mads_datasets import datatools

window_size = sunspotfactory.settings.window_size
print(f"windowsize: {window_size}")  # 26

idx = datatools.window(train, window_size)  # n_time=26 | n_window=2475 (2500-26+1)
trainset = train[idx]
idx = datatools.window(test, window_size)  # n_time=26 | n_window=798 (823-26+1)
testset = test[idx]
print("shape dataset/ train: "+str(trainset.shape)+", test: "+str(testset.shape))  # torch.Size([2475, 26]), torch.Size([798, 26])

# Ok, that is looking good. We squeezed out 2475 training examples out of this long, single timeseries.
# Let's make the data 3D, just as our timeseries model will need to have it. We just have one feature, so that is just a dimension of 1.

# Custom dataset
trainset = trainset[...,None]
testset = testset[..., None]
print("shape custom dataset/ train: "+str(trainset.shape)+", test: "+str(testset.shape))  # torch.Size([2475, 26, 1]), torch.Size([798, 26, 1])

# We can implement a Dataset in PyTorch. 
# According to the [documentation](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)
# > A custom Dataset class must implement three functions: `__init__`, ` __len__`, and `__getitem__`

# The functions with the double underscores are called "dunder" function. 
# The `__len__` function is the function that is called when you do `len(object)`, the `__getitem__` function is called when you do `object[idx]` and `idx` is passed as an argument.

# We can inherit the `Dataset` class, and if we implement a `__len__` and `__getitem__` function, we are good.

from mads_datasets.base import DatasetProtocol
from typing import Tuple
Tensor = torch.Tensor

class SunspotDataset(DatasetProtocol):
    def __init__(self, data: Tensor, horizon: int) -> None:
        self.data = data
        self.size = len(data)
        self.horizon = horizon

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # get a single item
        item = self.data[idx]
        # slice off the horizon
        x = item[:-self.horizon,:]
        y = item[-self.horizon:,:].squeeze(-1) # squeeze will remove the last dimension if possible.
        return x, y

datashape = (100, 10, 2)  # 100 samples, time steps 10, 2 features
dummydata = torch.randint(0, 10, datashape)
dummydataset = SunspotDataset(dummydata, horizon=2)
len(dummydataset) # uses the __len__ method
print("length dummydataset: "+ str(len(dummydataset)))  # 100

x, y = dummydataset[0] # uses the __getitem__ method
print("x.shape, y.shape: "+str(x.shape)+", "+str(y.shape)) # no squeeze because we have dimension 2 as the last dimension

# ---
# Explanation of slicing in __getitem__
# x = item[:-self.horizon, :]
# Take all but the last horizon time steps.
# If T = 10 and horizon = 2:
# item[:-2, :] → time steps 0..7 → shape (8, 2)
#
# Take the last horizon time steps:
# item[-2:, :] → time steps 8 and 9 → shape (2, 2)
# .squeeze(-1) tries to remove the last dimension if it has size 1.
# Here, last dimension size is 2 (because F = 2), so nothing is removed:
# y.shape stays (2, 2).
# ---

# Apply to trainset
trainset.shape

horizon = 3
traindataset = SunspotDataset(trainset, horizon=horizon)
testdataset = SunspotDataset(testset, horizon=horizon)
print("traindataset.data.shape: "+str(traindataset.data.shape))  # torch.Size([2475, 26, 1])
print("testdataset.data.shape: "+str(testdataset.data.shape))    # torch.Size([798, 26, 1])

# Dataloader
sunspotfactory.settings

from mltrainer.preprocessors import BasePreprocessor
preprocessor = BasePreprocessor()

streamers = sunspotfactory.create_datastreamer(batchsize=32, preprocessor=preprocessor)
train = streamers["train"]
valid = streamers["valid"]
trainstreamer = train.stream()
validstreamer = valid.stream()

x, y = next(iter(trainstreamer))
type(x), len(x), x[0].shape, type(y)  # (torch.Tensor, 32, torch.Size([23, 1]), torch.Tensor)
print("x.shape, y.shape")
print(x.shape, y.shape)               # torch.Size([32, 23, 1]), torch.Size([32, 3])
print("len(train), len(valid)")
print(len(train), len(valid))         # 82 20

# ---
# Explanation of len(train), len(valid):
# The exact numbers (82, 20) come from how the dataset factory internally:
# Builds its own windowed dataset,
# Splits into train/valid,
# Decides how many batches per epoch to use (usually based on dataset size and batchsize, sometimes with rounding / drop-last / specific settings).
# You don’t see that logic in your script because it lives inside the mads_datasets / mltrainer internals.
# However:
# - 82 x 32 = 2624 training samples per epoch (close to the 2475 available samples)
# - 20 x 32 = 640 validation samples per epoch (close to the 798 available samples)
# ---

# === NAIVE MODELS ===
# The most basic prediction is a Naive model. 

# - What is a naive prediction of the temperature for tomorrow? Well, the temperature of today.
# - The stock market for tomorrow? Same as today.
# - Bitcoin for march 16th of 2028? Same as march 15th 2028.

# In general, this is a super simple model without any parameters but often much better than you would expect, because the temparature of today will be close to the temperature of tomorrow.

# This gives rise to the MASE, the Mean Absolute Scaled Error: It is the mean absolute error of the forecast values (the actual model), but scaled by the mean absolute error of the in-sample one-step naive forecast (the naive model).
# This takes a moment in time t, and calculates the difference with one step back t-1.
# In other words, the MASE compares the MAE of your actual model to the MAE of the naive model. 
# E.g. if the error of your model is 1, and the error of the naive model is 10, your MASE is 0.1, meaning you are much better than the naive prediction.
# MASE values above 1 are really bad, because it means the naive function outperformed your actual model, everything below (ideally close to 0) is an improvement.
# We will implement this ourselves. First the naive prediction:

Tensor = torch.Tensor

def naivepredict(x: Tensor, horizon: int) -> tuple[Tensor, Tensor]:
    assert horizon > 0
    yhat = x[...,-horizon:, :].squeeze(-1)
    return yhat

yhat = naivepredict(x, horizon)  # this should be the same horizon you used to create the dataset
print("yhat.shape: "+str(yhat.shape))  # torch.Size([32, 3])

# With the naive prediction, we can calculate the MAE

def mae(y: np.ndarray, yhat: np.ndarray) -> float:  # Mean Absolute Error, which is average of |y - yhat|
    return np.mean(np.abs(y-yhat))

print("mae: "+str(mae(y.detach().numpy(), yhat.detach().numpy())))  # 0.058591787

# This error is not deterministic, because of the shuffling in the dataset.
# Now, we want to calculate the naive MAE for every batch, to get the average, expected MAE for the naive prediction

def naivenorm(train , horizon: int):
    elist = []  # list of errors
    streamer = train.stream()

    for _ in range(len(train)):  # len(train) is 82
        x, y = next(iter(streamer))
        yhat = naivepredict(x, horizon)
        e = mae(y.numpy(), yhat.numpy())  # MEA of this batch
        elist.append(e)
    return torch.mean(torch.tensor(elist))  # torch.tensor(elist) -> 1D tensor of shape (len(train),) with all batch MAEs.
                                            # torch.mean(...) -> overall average MAE of the naive predictor on the training stream.
                                            # returns a torch scalar (e.g. tensor(0.0565)).

print("naivenorm(train, 1): "+str(naivenorm(train, 1)))  # tensor(0.0565)

# This means, a MAE close to this number is not impressive at all. We need to get below this if we want to go beyond the naive prediction.

# We can wrap everything in a class, so it is neatly organized.
# This example also illustrates two other dunder methods, `__repr__` and `__call__`

from mads_datasets.base import BaseDatastreamer
Tensor = torch.Tensor
class MASE:
    def __init__(self, train: BaseDatastreamer, horizon: int):
        self.scale = self.naivenorm(train, horizon)

    def __repr__(self) -> str:
        # __repr__ is what is returned if you print the class to the screen
        return f"MASE(scale={self.scale:.3f})"

    def naivenorm(self, train: BaseDatastreamer, horizon: int) -> Tensor:
        elist = []
        streamer = train.stream()
        for _ in range(len(train)):
            x, y = next(iter(streamer))
            yhat = self.naivepredict(x, horizon)
            e = self.mae(y.numpy(), yhat.numpy())
            elist.append(e)
        return torch.mean(torch.tensor(elist))

    def naivepredict(self, x: Tensor, horizon: int) -> Tuple[Tensor, Tensor]:
        assert horizon > 0
        yhat = x[...,-horizon:, :].squeeze(-1)
        return yhat

    def mae(self, y: np.ndarray, yhat: np.ndarray) -> float:
        return np.mean(np.abs(y-yhat))

    def __call__(self, y: np.ndarray, yhat: np.ndarray) -> float:
        # __call__ lets us call the class as a function
        return self.mae(y, yhat) / self.scale

mase = MASE(train, horizon)
print("mase using __repr__: "+str(mase))  # MASE(scale=0.062)

mase(y.numpy(), yhat.numpy()) # __call__ in action
print("mase using __call__: "+str(mase(y.numpy(), yhat.numpy())))  # 0.9451

# ---
# Explanation of MASE class
# __init__ – initializes the class, calculates the scaling factor based on the training data.
# __repr__ – returns a string representation of the class instance for easy debugging.
# naivenorm – computes the average MAE of the naive predictor over the training data.
# naivepredict – generates naive predictions by taking the last observed values.
# mae – calculates the Mean Absolute Error between actual and predicted values.
# __call__ – allows the class instance to be called like a function to compute the MASE | Calculates: (model’s MAE on that batch ≈ 0.0586 / naive MAE on training data ≈ 0.0615) = MASE ≈ 0.9451
#
# Interpretation:
# MASE < 1 → model better than naive (smaller error than baseline)
# MASE = 1 → model same as naive
# MASE > 1 → model worse than naive
#
# So 0.9451 means: On this batch, your model is about 5.5% better (lower MAE) than the naive predictor.
# ---

# ---
# Explanation of how the 0.0615 is calculated:
# 1. We loop over 82 iterations as defined by len(train).
# 2. Each e is the mean absolute error over 32 × 3 = 96 scalar targets (3 per sample in the batch).
# 3. For each sample in the batch, the naive predictor takes the last 3 observed values in x and uses them as predictions for the next 3 time steps, giving a 3-element prediction vector yhat[i].
#    For that sample we compute 3 absolute errors (y[i, j] - yhat[i, j]), one per horizon step.
# 4. For each batch, y has 3 targets per sample -> 32 × 3 = 96 targets. We compute the absolute error between each y[i, j] and yhat[i, j], and mae averages those 96 errors into a single scalar e for that batch.
# 5. These 82 batch MAEs are then averaged to get the final naive norm (the scale shown in MASE(scale=0.0615)).
# So the final scale is based on about 82 × 32 × 3 = 7,872 errors, all of the form y[i, j] - yhat[i, j].
# ---

# === LINEAIR MODEL ===
# Ok, we have everything in place. Let's start with a slightly more advanced model than a naive one.
# We start with a linear baselinemodel, of just a single `Linear` layer with one unit. This is equal to a linear model.

# **NOTE** A Neural Network (consisting of Linear layers AND activation layers) is NOT a simple model; if your model is deep & wide enough, this is a universal function approximator!
# That means a neural network is capable, in theory, of approximating EVERY function.
# In this case we use a single layer, with a minimum amount of units; that's why this is a simple model!

# Check out the sourcecode (tip: VS code let's you do so easily by pressing F12, or by right-clicking the mouse on a function and selecting `Go to Definition`)

from mltrainer import rnn_models, metrics
observations = window_size - horizon
model = rnn_models.BaseModel(observations=observations, horizon=horizon)
x, y = next(iter(trainstreamer))
print("x.shape, y.shape")
print(x.shape, y.shape)  # torch.Size([32, 23, 1]) torch.Size([32, 3])

# Random model
yhat = model(x)
print("yhat.shape: "+str(yhat.shape))  # torch.Size([32, 3])

mase = metrics.MASE(train, horizon)
print("mase: "+str(mase(y.detach().numpy(), yhat.detach().numpy())))  # 2.139024496078491

# Oh boy! Untrained, this is actually much much worse than a naive model... Let's start training...
# First, we add a regular MAE too.

mae = metrics.MAE()
print("mae: "+str(mae(yhat.detach().numpy(), y.detach().numpy())))  # 0.13276931643486023

from loguru import logger
import sys
logger.remove()
logger.add(sys.stderr, level="SUCCESS")

from mltrainer import TrainerSettings, ReportTypes
from pathlib import Path

log_dir = Path("logs/rnn/").resolve()
settings = TrainerSettings(
    epochs=100,
    metrics=[mase, mae],
    logdir=log_dir,
    train_steps=len(train),
    valid_steps=len(valid),
    reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.TOML],
    scheduler_kwargs = {"factor": 0.5, "patience": 5},
    earlystop_kwargs = {
        "save": False,
        "verbose": True,
        "patience": 10,
    }
)

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

device = "cpu" # faster than mps in this case

from mltrainer import Trainer

observations = window_size - horizon
model = rnn_models.BaseModel(
    observations=observations,
    horizon=horizon,
)

# ---
# Explanation of observations and horizon
# - observations is the number of past time steps you feed into the model as input.
# - horizon is the number of future time steps the model should predict.
# So BaseModel is something like: take observations past points → output horizon future points.
# Why is this called a linear model?
# Typically in these tutorials, BaseModel is implemented (conceptually) like this:
# self.linear = torch.nn.Linear(in_features=observations, out_features=horizon)
# def forward(self, x):
#     # x shape: (batch, observations)
#     return self.linear(x)
# Key point: the mapping from input to output is just a linear layer: yhat = Wx + b
# There are no nonlinear activation functions (like ReLU, tanh, etc.) after that.
# Because of that, the model is linear in its inputs -> it’s basically multivariate linear regression in PyTorch form.
# So even though it lives in something called rnn_models, this specific BaseModel is a simple linear mapping from past values -> future values, hence “linear model”.

trainer = Trainer(
    model=model,
    settings=settings,
    loss_fn=torch.nn.MSELoss(),
    optimizer=torch.optim.Adam,
    traindataloader=trainstreamer,
    validdataloader=validstreamer,
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    device=device,
    )
trainer.loop()

# If you have a look at Tensorboard, you can see the model is learning!

x, y = next(iter(validstreamer))
yhat = model(x)
mae(y.detach().numpy(), yhat.detach().numpy()), mase(y.detach().numpy(), yhat.detach().numpy())

# ---
# Explanation of mea
# MSE  is standard for regression problems and time-series forecasting where the target is a real value.
# So loss_fn is telling the trainer: “When training, measure how wrong the predictions are using MSE.”

# And, yes, the MASE dropped below 1! This is a good start.

# Let's visualize this.
k = 4
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
axs = ax.ravel()

timesteps = x.shape[1]
idx0 = [*range(timesteps)]
idx1 = [(i + timesteps) for i in range(horizon)]
for i in range(k):
    axs[i].plot(idx0, x[i])
    axs[i].plot(idx1, yhat[i].detach(), color="red", label="predict")
    axs[i].plot(idx1, y[i], color="green", label="actual")
handles, labels = axs[i].get_legend_handles_labels()
fig.legend(handles, labels)
plt.show()

# Well, it is not perfect, obviously. But it does seem to capture the trend more often that random.

# === SIMPLE RNN ===
# Let's invoke the basic RNN without gates.
# Input size is 1, because we have 1 feature. We follow [the batch_first convention](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html).
# Hidden size is the inner dimension of the RNN, and thus the dimensionality of the output.
# What would happen if we feed this data with dimensions `(batch, sequence_length)`?
x = torch.randn((32, 25))
rnn = torch.nn.RNN(input_size=1, hidden_size=1, batch_first=True)
try:
    yhat = rnn(x)
except RuntimeError as e:
    print(e)

# Note how it expects 3 dimensions: `(batch, sequence_length, features)`, even though we have just one feature.

x, y = next(iter(trainstreamer))
print("x.shape, y.shape")
print(x.shape, y.shape)

out, hidden = rnn(x)
print("out.shape, hidden.shape")
print(out.shape, hidden.shape)

# This works as expected. Let's increase the hidden dimension (why would you do that?)

rnn = torch.nn.RNN(input_size=1, hidden_size=10, batch_first=True)
out, hidden = rnn(x)
print("out.shape, hidden.shape #2")
print(out.shape, hidden.shape)

# === Visualiszatrion of RNN outputs UNTRAINED (added by AvT) ===
import matplotlib.pyplot as plt
import torch

# Get one batch
x, y = next(iter(trainstreamer))
print("x.shape, y.shape")
print(x.shape, y.shape)   # (batch, seq_len, 1)

# --- First RNN: hidden_size = 1 ---
rnn1 = torch.nn.RNN(input_size=1, hidden_size=1, batch_first=True)
out1, hidden1 = rnn1(x)
print("out1.shape, hidden1.shape")
print(out1.shape, hidden1.shape)  # (batch, seq_len, 1), (1, batch, 1)

# --- Second RNN: hidden_size = 10 ---
rnn2 = torch.nn.RNN(input_size=1, hidden_size=10, batch_first=True)
out2, hidden2 = rnn2(x)
print("out2.shape, hidden2.shape")
print(out2.shape, hidden2.shape)  # (batch, seq_len, 10), (1, batch, 10)

# === Plot both results in ONE plot ===
# Choose one example from the batch
i = 0
seq_len = x.shape[1]
timesteps = range(seq_len)

# hidden_size = 1: (batch, seq_len, 1) -> take sample 0, feature 0
series1 = out1[i, :, 0].detach().numpy()              # shape: (seq_len,)

# hidden_size = 10: (batch, seq_len, 10) -> take sample 0
series2 = out2[i].detach().numpy()                    # shape: (seq_len, 10)

plt.figure(figsize=(8, 4))

# Blue line: hidden_size = 1
plt.plot(timesteps, series1, label="hidden_size = 1", color="blue")

# Orange dots: all 10 hidden units for hidden_size = 10
for j in range(series2.shape[1]):  # j = 0..9
    plt.scatter(
        timesteps,
        series2[:, j],
        color="orange",
        s=20,
        alpha=0.7,
        label="hidden_size = 10 (10 units)" if j == 0 else None,
    )

plt.xlabel("timestep")
plt.ylabel("RNN hidden state")
plt.title("RNN outputs: hidden_size = 1 vs hidden_size = 10")
plt.legend()
plt.tight_layout()
plt.show()
# This visualization helps to see how the RNN outputs change with different hidden sizes. RNN outputs are the hidden states.
# For hidden_size=1, there's only one output per timestep (blue). For hidden_size=10, there are 10 outputs per timestep (orange).
# You see that weights and biases are used, however there are randomly initialized, so the outputs look random and are not trained yet (BaseRNN) is not called yet.

# So, let's wrap this up in a model and treain the RNN. Study the implementation in `rnn_models.BaseRNN`.
# Do you understand what the linear layer does? What would happen if you remove it?

rnn3 = torch.nn.RNN(
    input_size=1,
    hidden_size=10,
    num_layers=1,
    batch_first=True)
out3, hidden3 = rnn3(x)
out3.shape, hidden3.shape
print("out3.shape, hidden3.shape #3")
print(out3.shape, hidden3.shape)  # torch.Size([32, 23, 10]) torch.Size([1, 32, 10])

# To get the output of the last timestep, we need to throw away the rest.
out3 = out3[:,-1,:]
out3.shape
print("out3.shape: "+str(out3.shape))  # out3.shape: torch.Size([32, 10])

from mltrainer import metrics, rnn_models

mase = metrics.MASE(train, horizon)
mae = metrics.MAE()

settings.epochs = 10
device = "cpu" # still faster than mps

observations = window_size - horizon
model = rnn_models.BaseRNN(
    input_size=1,
    hidden_size=10,
    num_layers=3,
    horizon=horizon,
    )

trainer = Trainer(
    model=model,
    settings=settings,
    loss_fn=torch.nn.MSELoss(),
    optimizer=torch.optim.Adam,
    traindataloader=trainstreamer,
    validdataloader=validstreamer,
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    device=device,
    )
trainer.loop()

# === Visualiszatrion of RNN outputs TRAINED (added by AvT) ===
import matplotlib.pyplot as plt
import torch

# Make sure the model is in eval mode (no dropout, etc.)
model.eval()

# Take one fixed batch to visualize
x_vis, y_vis = next(iter(trainstreamer))
print("x_vis.shape, y_vis.shape:", x_vis.shape, y_vis.shape)  # x_vis.shape, y_vis.shape: torch.Size([32, 23, 1]) torch.Size([32, 3])

with torch.no_grad():
    # Pass through the trained RNN *inside* BaseRNN, not through the full model
    out_vis, hidden_vis = model.rnn(x_vis)
    # out_vis: (batch, seq_len, hidden_size=10)

# Choose one example from the batch
i = 0
seq_len = x_vis.shape[1]
timesteps = range(seq_len)

# Hidden states for sample i: shape (seq_len, 10)
hidden_seq = out_vis[i].detach().numpy()

plt.figure(figsize=(8, 4))

# Plot all 10 hidden units as orange dots
for j in range(hidden_seq.shape[1]):  # j = 0..9
    plt.scatter(
        timesteps,
        hidden_seq[:, j],
        color="orange",
        s=20,
        alpha=0.7,
        label="trained hidden (10 units)" if j == 0 else None,
    )

plt.xlabel("timestep")
plt.ylabel("hidden state value")
plt.title("Trained RNN hidden states (hidden_size=10)")
plt.legend()
plt.tight_layout()
plt.show()
# Same as above, but now with a trained RNN (1 batch). You can see the hidden states now have learned patterns, unlike the random untrained RNN.

# === Increase num_layers to 3 ===
# And finally let's add multiple stacked layers of RNN (can you visualize how this would work?)
rnn4 = torch.nn.RNN(
    input_size=1,
    hidden_size=10,
    num_layers=3,
    batch_first=True)
out4, hidden4 = rnn4(x)
out4.shape, hidden4.shape
print("out4.shape, hidden4.shape #4")
print(out4.shape, hidden4.shape)  # torch.Size([32, 23, 10]) torch.Size([3, 32, 10])

# To get the output of the last timestep, we need to throw away the rest.
out4 = out4[:,-1,:]
out4.shape
print("out4.shape: "+str(out4.shape))  # out4.shape: torch.Size([32, 10])

# ---
# Explanation of adding more layers (num_layers).
# Having only 1 layer:
# - Input: shape (batch=32, seq_len=23, input_size=1)
# - Output out3: shape (32, 23, 10) – hidden state of layer 1 at every timestep.
# - Hidden3: shape (1, 32, 10) – last hidden state of layer 1 for every sequence in batch.
#
# Having 3 layers:
# - Input: shape (batch=32, seq_len=23, input_size=1); remains unchanged.
# - Output out3: shape (32, 23, 10) – hidden state of layer 1 at every timestep; remains unchanged.
# - Hidden3: shape (3, 32, 10) – last hidden state of layer 1 for every sequence in batch.
#
# Key things:
# The shape of out does not change – it is always the output of the top (last) layer for every timestep.
# hidden now has one entry per layer:
# - hidden[0, :, :] → last hidden state of layer 1
# - hidden[1, :, :] → last hidden state of layer 2
# - hidden[2, :, :] → last hidden state of layer 3
# So num_layers only changes the depth of the stack, not the output shape.
#
# Conceptual difference: more layers vs larger hidden_size
# - hidden_size -> width of each layer (dimension of the hidden state vector per timestep).
# - num_layers -> depth of the network (how many RNN layers you stack on top of each other).
# a) hidden_size ↑ (e.g. 10 -> 50)
#    - Each timestep’s hidden state has more components (more capacity to encode info).
#    - Still only one layer transforming input over time.
# b) num_layers (e.g. 1 -> 3)
#    - You now have a pipeline of RNNs:
#    - Layer 1 learns lower-level temporal patterns.
#    - Higher layers can learn more abstract temporal features built on top of those.
# Analogy:
# - hidden_size -> how many numbers you can store in your “memory” at each step.
# - num_layers -> how many “processing stages” the sequence goes through at each time step.
#
# What changes during training?
# a) All 3 layers get their own weights & biases:
#    - Layer 1: weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
#    - Layer 2: weight_ih_l1, weight_hh_l1, ...
#    - Layer 3: weight_ih_l2, weight_hh_l2, ...
# b) During training, gradients flow through all layers, so all layers are trained to cooperate:
#    - lower layers extract basic time patterns,
#    - higher layers refine them towards what’s needed to predict your horizon.
# ---


from mltrainer import metrics, rnn_models

mase = metrics.MASE(train, horizon)
mae = metrics.MAE()

settings.epochs = 10
device = "cpu" # still faster than mps

observations = window_size - horizon
model = rnn_models.BaseRNN(
    input_size=1,
    hidden_size=10,
    num_layers=3,
    horizon=horizon,
    )

trainer = Trainer(
    model=model,
    settings=settings,
    loss_fn=torch.nn.MSELoss(),
    optimizer=torch.optim.Adam,
    traindataloader=trainstreamer,
    validdataloader=validstreamer,
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    device=device,
    )
trainer.loop()

# === Visualiszatrion of RNN outputs TRAINED layer 1 and layer 3 (added by AvT) ===
import matplotlib.pyplot as plt
import torch

# Make sure we're in eval mode
model.eval()

# Take one batch
x_vis, y_vis = next(iter(trainstreamer))
print("x_vis.shape, y_vis.shape:", x_vis.shape, y_vis.shape)  # x_vis.shape, y_vis.shape: torch.Size([32, 23, 1]) torch.Size([32, 3])

with torch.no_grad():
    # Pass through the trained 3-layer RNN inside BaseRNN
    out_top, hidden_all = model.rnn(x_vis)
    # out_top: (batch, seq_len, hidden_size=10)
    # hidden_all: (num_layers=3, batch, hidden_size=10)

    # === Build a separate 1-layer RNN that uses the trained layer-1 weights ===
    rnn_layer1 = torch.nn.RNN(
        input_size=1,
        hidden_size=10,
        num_layers=1,
        batch_first=True,
    )

    # Copy layer 0 (first layer) weights/biases from the trained stacked RNN
    rnn_layer1.weight_ih_l0.data = model.rnn.weight_ih_l0.data.clone()
    rnn_layer1.weight_hh_l0.data = model.rnn.weight_hh_l0.data.clone()
    rnn_layer1.bias_ih_l0.data   = model.rnn.bias_ih_l0.data.clone()
    rnn_layer1.bias_hh_l0.data   = model.rnn.bias_hh_l0.data.clone()

    out_l1, hidden_l1 = rnn_layer1(x_vis)   # (batch, seq_len, 10)

# === Plot layer 1 vs layer 3 hidden trajectories (10 dots each) ===

# Choose one sequence from the batch
i = 0
seq_len = x_vis.shape[1]
timesteps = range(seq_len)

hidden_l1 = out_l1[i].detach().numpy()    # (seq_len, 10)
hidden_l3 = out_top[i].detach().numpy()   # (seq_len, 10)

plt.figure(figsize=(8, 4))

# Layer 1: 10 orange dots per timestep
for j in range(hidden_l1.shape[1]):  # j = 0..9
    plt.scatter(
        timesteps,
        hidden_l1[:, j],
        color="orange",
        s=20,
        alpha=0.7,
        label="Layer 1 (10 units)" if j == 0 else None,
    )

# Layer 3: 10 yellow dots per timestep
for j in range(hidden_l3.shape[1]):  # j = 0..9
    plt.scatter(
        timesteps,
        hidden_l3[:, j],
        color="yellow",
        s=20,
        alpha=0.7,
        marker="x",
        label="Layer 3 (10 units)" if j == 0 else None,
    )

plt.xlabel("timestep")
plt.ylabel("hidden state value")
plt.title("Hidden states: Layer 1 (orange) vs Layer 3 (yellow)")
plt.legend()
plt.tight_layout()
plt.show()

print("\nTraining finished.")
print("To view TensorBoard, run this in a terminal:")
print(f"  tensorboard --logdir {log_dir} --port 6006")
print("Then open your browser at http://localhost:6006\n")


# Does this improve over the linear model? -> AvT: No
# By how much? Is there a difference between loss and MASE? 
#
# Use the tensorboard to find this out!
# What does it mean? -> AvT: RNN are overfitting
# Can you improve the model by tweaking it?
# AvT: If you wanted to make RNNs truly competitive here, you’d start thinking about:
#      - regularization (dropout, weight decay),
#      - gradient clipping,
#      - tuning learning rate,
#      - maybe using LSTM/GRU,
#      - or just accepting: “for this problem, the linear model is enough.”