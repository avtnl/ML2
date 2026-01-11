import torch.nn as nn
import torch
from torch import Tensor
from dataclasses import dataclass
from pathlib import Path
from torch import optim
import mlflow
from datetime import datetime
from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer import TrainerSettings, Trainer, ReportTypes
from mltrainer.metrics import Accuracy
from mltrainer.preprocessors import PaddedPreprocessor
preprocessor = PaddedPreprocessor()

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using MPS")
elif torch.cuda.is_available():
    device = "cuda:0"
    print("using cuda")
else:
    device = "cpu"
    print("using cpu")

gesturesdatasetfactory = DatasetFactoryProvider.create_factory(DatasetType.GESTURES)
streamers = gesturesdatasetfactory.create_datastreamer(batchsize=32, preprocessor=preprocessor)
train = streamers["train"]
valid = streamers["valid"]
print("len(train) | len(valid):", len(train), len(valid))

trainstreamer = train.stream()
validstreamer = valid.stream()
x, y = next(iter(trainstreamer))
print("x.shape | y:", x.shape, y)

accuracy = Accuracy()

loss_fn = torch.nn.CrossEntropyLoss()
#loss = loss_fn(yhat, y)
#print("loss:", loss)

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
    hidden_size=128,
    num_layers=4,
    output_size=20,
    dropout=0.15,
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
        hidden_size=128,
        num_layers=4,
        output_size=20,
        dropout=0.15,
    )

    model = GRUmodel(
        config=config,
    )

    settings = TrainerSettings(
        epochs=25, # increase this to about 100 for training
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