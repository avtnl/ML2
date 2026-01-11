# hypertune2.py — FINAL, WORKS PERFECTLY (Ray 2.10+, Windows, 2025)
from pathlib import Path
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from filelock import FileLock
import torch
import torch.nn as nn

NUM_SAMPLES = 50
MAX_EPOCHS = 10


def train(config):
    from loguru import logger
    from mads_datasets import DatasetFactoryProvider, DatasetType
    from mltrainer.preprocessors import PaddedPreprocessor
    from mltrainer.metrics import Accuracy
    import importlib

    rnn_models = importlib.import_module("mltrainer.rnn_models")

    data_dir = Path(config["data_dir"])
    factory = DatasetFactoryProvider.create_factory(DatasetType.GESTURES)
    preprocessor = PaddedPreprocessor()

    with FileLock(data_dir / ".lock"):
        streamers = factory.create_datastreamer(batchsize=32, preprocessor=preprocessor)
        train_stream = streamers["train"].stream()
        valid_stream = streamers["valid"].stream()

    model = rnn_models.GRUmodel(rnn_models.ModelConfig(
        input_size=config["input_size"],
        output_size=config["output_size"],
        hidden_size=config["hidden_size"],
        dropout=config["dropout"],
        num_layers=config["num_layers"],
    ))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    metric = Accuracy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_steps = len(streamers["train"])
    valid_steps = len(streamers["valid"])

    for epoch in range(MAX_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for _ in range(train_steps):
            x, y = next(train_stream)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for _ in range(valid_steps):
                x, y = next(valid_stream)
                x, y = x.to(device), y.to(device)
                yhat = model(x)
                loss = loss_fn(yhat, y)
                valid_loss += loss.item()
                pred = yhat.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        valid_loss /= valid_steps
        accuracy = correct / total

        # CORRECT 2025 WAY — DICT ARGUMENT
        tune.report({"test_loss": valid_loss, "Accuracy": accuracy})

if __name__ == "__main__":
    from mads_datasets import DatasetFactoryProvider, DatasetType
    from mltrainer.preprocessors import PaddedPreprocessor

    print("Pre-loading dataset once...")
    factory = DatasetFactoryProvider.create_factory(DatasetType.GESTURES)
    _ = factory.create_datastreamer(batchsize=32, preprocessor=PaddedPreprocessor())
    print("Dataset ready!\n")

    ray.init()

    def short_name(trial):
        return f"trial_{trial.trial_id[:8]}"

    data_dir = Path("data/raw/gestures/gestures-dataset").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "data_dir": str(data_dir),
        "input_size": 3,
        "output_size": 20,
        "hidden_size": tune.randint(16, 128),
        "dropout": tune.uniform(0.0, 0.3),
        "num_layers": tune.randint(2, 5),
    }

    analysis = tune.run(
        train,
        config=config,
        metric="test_loss",
        mode="min",
        num_samples=NUM_SAMPLES,
        search_alg=HyperOptSearch(),
        scheduler=AsyncHyperBandScheduler(
            time_attr="training_iteration",
            grace_period=1,
            reduction_factor=3,
            max_t=MAX_EPOCHS,
        ),
        progress_reporter=CLIReporter(metric_columns=["test_loss", "Accuracy"]),
        storage_path=str(Path("logs/ray").resolve()),
        trial_dirname_creator=short_name,
        verbose=1,
    )

    print("\nBEST CONFIGURATION FOUND:")
    best = analysis.get_best_config(metric="test_loss", mode="min")
    print(best)
    print(f"Best final accuracy: {analysis.best_result['Accuracy']:.4f}")
    ray.shutdown()