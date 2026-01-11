import time
import numpy as np
from plotly import graph_objects as go
import matplotlib.pyplot as plt

import os
# CRITICAL: Set BEFORE ANY Ray imports to disable auto-loggers (including TBX)
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

# Optional: Silence the GPU warning
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

from loguru import logger

import ray
from ray import tune

import torch
from torch import optim
from torch import nn

from pathlib import Path

import time
import numpy as np

from ray.tune.logger import CSVLoggerCallback, JsonLoggerCallback

plt.style.use('dark_background')
# import plotly.io as pio
# pio.renderers.default = 'plotly_mimetype+notebook'

import visualize
DELETE = True # to delete the tunedir at the end of the notebook
start = time.time()

MAX_EPOCHS = 10
N_EXPERIMENTS = 18

timer = {}
best_config = {}

def train_fn(model, trainstreamer, lossfn, optimizer, steps):
    model.train()
    train_loss: float = 0.0
    for _ in range(steps):
        x, y = next(trainstreamer)
        optimizer.zero_grad()
        yhat = model(x)
        loss = lossfn(yhat, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss

def validate(model, validstreamer, lossfn, metric, steps):
    model.eval()
    valid_loss: float = 0.0
    acc: float = 0.0
    with torch.no_grad():
        for _ in range(steps):
            x, y = next(validstreamer)
            yhat = model(x)
            loss = lossfn(yhat, y)
            valid_loss += loss.item()
            acc += metric(y, yhat)
    acc /= steps
    return valid_loss, acc

def get_data(tune_dir):
    from filelock import FileLock
    from mads_datasets import DatasetFactoryProvider, DatasetType
    from mltrainer.preprocessors import PaddedPreprocessor
    from mads_datasets import DatasetFactoryProvider, DatasetType
    with FileLock(tune_dir / ".lock"):
        # we lock the datadir to avoid parallel instances trying to
        # access the datadir
        preprocessor = PaddedPreprocessor()
        gesturesdatasetfactory = DatasetFactoryProvider.create_factory(DatasetType.GESTURES)
        streamers = gesturesdatasetfactory.create_datastreamer(
            batchsize=32, preprocessor=preprocessor
        )
        train = streamers["train"]
        valid = streamers["valid"]
    return train, valid

class GRUmodel(nn.Module):
    def __init__(
        self,
        config: dict,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=config["input_size"],
            hidden_size=int(config["hidden_size"]),
            dropout=config["dropout"],
            batch_first=True,
            num_layers=int(config["num_layers"]),
        )
        self.linear = nn.Linear(int(config["hidden_size"]), config["output_size"])

    def forward(self, x):
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat

def tune_model(config: dict):
    from mltrainer.metrics import Accuracy

    # load data
    train, valid = get_data(config["tune_dir"])
    trainsteps = len(train)
    validsteps = len(valid)
    trainstreamer = train.stream()
    validstreamer = valid.stream()

    # create model with config
    model = GRUmodel(config)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    metric = Accuracy()

    for _ in range(config["epochs"]):
        # train and validate
        train_loss = train_fn(model, trainstreamer, loss_fn, optimizer, trainsteps)
        valid_loss, accuracy = validate(model, validstreamer, loss_fn, metric, validsteps)

        # report to ray
        ray.tune.report({
            "valid_loss": valid_loss / validsteps,
            "train_loss": train_loss / trainsteps,
            "accuracy" : accuracy,
            })

tune_dir = Path("logs/test/").resolve()
print("tune_dir:", tune_dir)
 
config = {
    "input_size": 3,
    "output_size": 20,
    "dropout": 0.1,
    "hidden_size": 64,
    "num_layers": 2,
    "epochs": MAX_EPOCHS,
    "tune_dir": tune_dir,
}
tune_model(config)

config = {
    "input_size": 3,
    "output_size": 20,
    "dropout": 0.05,
    "epochs": MAX_EPOCHS,
    "hidden_size": tune.randint(16, 512),
    "num_layers": tune.randint(1, 8),
    "tune_dir": tune_dir,
}

tic = time.time()
analysis = tune.run(
    tune_model,
    config=config,
    metric="valid_loss",
    mode="min",
    storage_path=str(tune_dir),
    num_samples=N_EXPERIMENTS,
    stop={"training_iteration": MAX_EPOCHS},
    verbose=1,
    callbacks=[
        CSVLoggerCallback(),     # writes progress.csv inside each trial folder
        JsonLoggerCallback(),    # writes params.json + result.json inside each trial folder
    ]
)

timer["ray_random"] = time.time() - tic
print("timer:", timer["ray_random"])  

def plot_contour(df, x, y, z, start=0.90, end=1.0, size=0.01):
    fig = go.Figure()
    # ... your existing code ...

    fig.update_layout(
        title="Contour Plot – Random Search (6 trials)",
        xaxis_title="Hidden Size",
        yaxis_title="Number of Layers",
        # ... rest unchanged
    )

    fig.show(renderer="browser")   # ← THIS LINE
    # or even simpler:
    # fig.write_html("contour_plot.html")  # creates a file you can double-click


def plot_contour(df, x, y, z, start=0.90, end=1.0, size=0.01):
    fig = go.Figure()

    fig.add_trace(
        go.Contour(
            z=df[z],
            x=df[x],
            y=df[y],
            contours=dict(
                coloring='heatmap',
                showlabels=True,  # show labels on contours
                start=start,       # start of the contour range
                end=end,          # end of the contour range
                size=size,
            ),
            colorscale="plasma",
            colorbar=dict(
                title='Accuracy'
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y],
            mode='markers',
            marker=dict(
                color='black',
                size=8,
                symbol='circle'
            ),
            customdata=df['accuracy'],  # Pass accuracy values for hover text
            hovertemplate=(
                'Hidden Size: %{x}<br>'
                'Number of Layers: %{y}<br>'
                'Accuracy: %{customdata:.4f}<extra></extra>'
            ),
            name='Data Points'
        )
    )

    fig.update_layout(
        title="Contour Plot",
        xaxis_title="Hidden Size",
        yaxis_title="Number of Layers",
        xaxis=dict(showgrid=False),  # Remove x-axis grid lines
        yaxis=dict(showgrid=False),
        plot_bgcolor='white',        # Set background color to white
        paper_bgcolor='white'
    )

    fig.show(renderer="browser")
plt.show()

all_results = analysis.results_df

random = analysis.results_df
plot_contour(random, "config/hidden_size", "config/num_layers", "accuracy")

best = analysis.get_best_config()
best["accuracy"] = analysis.best_result["accuracy"]
best_config["random"] = best

print("best:", best)

# columns = ["config/hidden_size", "config/num_layers", "accuracy"]
# visualize.parallel_plot(analysis, columns)

import plotly.express as px

print("\nGenerating parallel coordinates plot...")
df = analysis.results_df

# Extract the config columns properly for Plotly
df_plot = df.copy()
df_plot["hidden_size"] = df["config/hidden_size"]
df_plot["num_layers"]   = df["config/num_layers"]
df_plot["accuracy"]     = df["accuracy"]

fig = px.parallel_coordinates(
    df_plot,
    color="accuracy",
    dimensions=["hidden_size", "num_layers", "accuracy"],
    color_continuous_scale=px.colors.sequential.Plasma_r,
    title="Parallel Coordinates Plot – Random Search (6 trials)"
)
fig.update_layout(coloraxis_colorbar=dict(title="Accuracy"))
fig.show(renderer="browser")
fig.write_html("logs/test/parallel_coordinates.html")
print("Parallel coordinates plot saved to logs/test/parallel_coordinates.html")

# === GRID SEARCH ===
config = {
    "input_size": 3,
    "output_size": 20,
    "dropout": 0.1,
    "epochs": MAX_EPOCHS,
    "hidden_size": tune.grid_search([16, 32, 64, 128, 256, 512]),
    "num_layers": tune.grid_search([2, 4, 8]),
    "tune_dir": tune_dir,
}

tic = time.time()

analysis = tune.run(
    tune_model,
    config=config,
    metric="valid_loss",
    mode="min",
    storage_path=str(tune_dir),
    stop={"training_iteration": MAX_EPOCHS},
    verbose=1,
)

timer["ray_grid"] = time.time() - tic

best = analysis.get_best_config()
best["accuracy"] = analysis.best_result["accuracy"]
best_config["grid"] = best

import pandas as pd
all_results = pd.concat([all_results, analysis.results_df])

grid = analysis.results_df
plot_contour(grid, "config/hidden_size", "config/num_layers", "accuracy", start=0.7)

#visualize.parallel_plot(analysis, columns)
print("\nGenerating parallel coordinates plot...")
df = analysis.results_df

# Extract the config columns properly for Plotly
df_plot = df.copy()
df_plot["hidden_size"] = df["config/hidden_size"]
df_plot["num_layers"]   = df["config/num_layers"]
df_plot["accuracy"]     = df["accuracy"]

fig = px.parallel_coordinates(
    df_plot,
    color="accuracy",
    dimensions=["hidden_size", "num_layers", "accuracy"],
    color_continuous_scale=px.colors.sequential.Plasma_r,
    title="Parallel Coordinates Plot – Random Search (6 trials)"
)
fig.update_layout(coloraxis_colorbar=dict(title="Accuracy"))
fig.show(renderer="browser")
fig.write_html("logs/test/parallel_coordinates.html")
print("Parallel coordinates plot saved to logs/test/parallel_coordinates.html")

visualize.plot_timers(timer)

# === BAYES ===
from ray.tune.search.bayesopt import BayesOptSearch

bayesopt = BayesOptSearch(metric="valid_loss", mode="min")


config = {
    "input_size": 3,
    "output_size": 20,
    "dropout": 0.1,
    "epochs": MAX_EPOCHS,
    "hidden_size": tune.uniform(16, 512),
    "num_layers": tune.uniform(1, 8),
    "tune_dir": tune_dir,
}

tic = time.time()

analysis = tune.run(
    tune_model,
    config=config,
    metric="valid_loss",
    mode="min",
    storage_path=str(tune_dir),
    num_samples=N_EXPERIMENTS,
    stop={"training_iteration": MAX_EPOCHS},
    search_alg=bayesopt,
    verbose=1,
)

timer["ray_bayes"] = time.time() - tic

best = analysis.get_best_config()
best["accuracy"] = analysis.best_result["accuracy"]
best_config["bayes"] = best

print("best:", best)

all_results = pd.concat([all_results, analysis.results_df])

bayes = analysis.results_df
plot_contour(bayes, "config/hidden_size", "config/num_layers", "accuracy", start=0.7)

visualize.parallel_plot(analysis, columns)

visualize.plot_timers(timer)

import pandas as pd

pd.DataFrame.from_dict(best_config, orient="index")

# === HYPERBAND ===
from ray.tune.schedulers import AsyncHyperBandScheduler

scheduler = AsyncHyperBandScheduler(
    time_attr="training_iteration", grace_period=1, reduction_factor=3, max_t=MAX_EPOCHS
)

config = {
    "input_size": 3,
    "output_size": 20,
    "dropout": 0.1,
    "epochs": MAX_EPOCHS,
    "hidden_size": tune.randint(16, 512),
    "num_layers": tune.randint(1, 8),
    "tune_dir": tune_dir,
}

tic = time.time()
analysis = tune.run(
    tune_model,
    config=config,
    metric="valid_loss",
    mode="min",
    storage_path=str(tune_dir),
    num_samples=N_EXPERIMENTS,
    stop={"training_iteration": MAX_EPOCHS},
    scheduler=scheduler,
    verbose=1,
)

timer["ray_hyperband"] = time.time() - tic

best = analysis.get_best_config()
best["accuracy"] = analysis.best_result["accuracy"]
best_config["hyperband"] = best

all_results = pd.concat([all_results, analysis.results_df])

hyperband = analysis.results_df
plot_contour(hyperband, "config/hidden_size", "config/num_layers", "accuracy", start=0.7)

visualize.parallel_plot(analysis, columns)

visualize.plot_timers(timer)

pd.DataFrame.from_dict(best_config, orient="index")

# === HYPERBAYES ===
from ray.tune.search.hyperopt import HyperOptSearch
search = HyperOptSearch()

scheduler = AsyncHyperBandScheduler(
    time_attr="training_iteration", grace_period=1, reduction_factor=3, max_t=MAX_EPOCHS
)

config = {
    "input_size": 3,
    "output_size": 20,
    "dropout": 0.1,
    "epochs": MAX_EPOCHS,
    "hidden_size": tune.randint(16, 512),
    "num_layers": tune.randint(1, 8),
    "tune_dir": tune_dir,
}

tic = time.time()
analysis = tune.run(
    tune_model,
    config=config,
    metric="valid_loss",
    mode="min",
    storage_path=str(tune_dir),
    num_samples=N_EXPERIMENTS,
    stop={"training_iteration": MAX_EPOCHS},
    search_alg=search,
    scheduler=scheduler,
    verbose=1,
)

timer["ray_hyperopt"] = time.time() - tic
contour = all_results[all_results["training_iteration"] == MAX_EPOCHS]
plot_contour(contour, "config/hidden_size", "config/num_layers", "accuracy", start=0.8)

best = analysis.get_best_config()
best["accuracy"] = analysis.best_result["accuracy"]
best_config["hyperopt"] = best

all_results = pd.concat([all_results, analysis.results_df])

hyperbayes = analysis.results_df
plot_contour(hyperbayes, "config/hidden_size", "config/num_layers", "accuracy", start=0.3, size=0.05)

visualize.parallel_plot(analysis, columns)

visualize.plot_timers(timer)

pd.DataFrame.from_dict(best_config, orient="index")

contour = all_results[all_results["training_iteration"] == MAX_EPOCHS]
plot_contour(contour, "config/hidden_size", "config/num_layers", "accuracy", start=0.8)

stop = time.time() - start
print(stop)

if DELETE:
    import shutil
    shutil.rmtree(tune_dir)
