# Let's try to see how Attention makes a difference with the gestures dataset
import torch
from mads_datasets import datatools
from mltrainer import metrics, Trainer, rnn_models
from torch import optim

# Get the data
from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer.preprocessors import PaddedPreprocessor
preprocessor = PaddedPreprocessor()

gesturesdatasetfactory = DatasetFactoryProvider.create_factory(DatasetType.GESTURES)
streamers = gesturesdatasetfactory.create_datastreamer(batchsize=32, preprocessor=preprocessor)
train = streamers["train"]
valid = streamers["valid"]

# Set up loss and accuraccy
from pathlib import Path
accuracy = metrics.Accuracy()
loss_fn = torch.nn.CrossEntropyLoss()
log_dir = Path("modellogs/attention").resolve()

# Set up configuration
from mltrainer import TrainerSettings, ReportTypes

settings = TrainerSettings(
    epochs=3,
    metrics=[accuracy],
    logdir=log_dir,
    train_steps=len(train),
    valid_steps=len(valid),
    reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.TOML,],
    scheduler_kwargs={"factor": 0.5, "patience": 5},
    earlystop_kwargs=None
)
config = rnn_models.ModelConfig(
    input_size=3,
    hidden_size=128,
    dropout=0.06,
    num_layers=4,
    output_size=20,
)

grumodel = rnn_models.GRUmodel(config)
print(settings)

# epochs: 3
# metrics: [Accuracy]
# logdir: C:\Users\avtnl\Documents\HU\ML\MADS-MachineLearning-course\notebooks\5_attention\modellogs\attention
# train_steps: 81
# valid_steps: 20
# reporttypes: [<ReportTypes.TENSORBOARD: 'TENSORBOARD'>, <ReportTypes.TOML: 'TOML'>]
# optimizer_kwargs: {'lr': 0.001, 'weight_decay': 1e-05}
# scheduler_kwargs: {'factor': 0.5, 'patience': 5}
# earlystop_kwargs: None

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using MPS")
elif torch.cuda.is_available():
    device = "cuda:0"
    print("using cuda")
else:
    device = "cpu"
    print("using cpu")

grutrainer = Trainer(
    model=grumodel,
    settings=settings,
    loss_fn=loss_fn,
    optimizer=optim.Adam,
    traindataloader=train.stream(),
    validdataloader=valid.stream(),
    scheduler=optim.lr_scheduler.ReduceLROnPlateau,
    device=device,
    )

# GRU model like we have seen before
grutrainer.loop()

attentionmodel = rnn_models.AttentionGRU(config)

# Compare performance with an attention layer added to the model as below
attentiontrainer = Trainer(
    model=attentionmodel,
    settings=settings,
    loss_fn=loss_fn,
    optimizer=optim.Adam,
    traindataloader=train.stream(),
    validdataloader=valid.stream(),
    scheduler=optim.lr_scheduler.ReduceLROnPlateau,
    device=device,
    )

attentiontrainer.loop()

# Take aways:
# - loss is better in both train and test
# - accuracy is better
# - the model converges faster

