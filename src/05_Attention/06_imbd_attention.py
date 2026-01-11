from pathlib import Path

from mltrainer import Trainer, metrics
from mltrainer.rnn_models import NLPmodel, AttentionNLP

import torch
from torch.utils.data import DataLoader
from torch import optim


from mads_datasets import DatasetFactoryProvider, DatasetType

# Load the streamers from the datasetfactory
imdbdatasetfactory = DatasetFactoryProvider.create_factory(DatasetType.IMDB)
datasets = imdbdatasetfactory.create_dataset()

traindataset = datasets["train"]
print(imdbdatasetfactory.settings)

# dataset_url: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# filename: aclImdb_v1.tar.gz
# name: imdb
# unzip: True
# formats: [<FileTypes.TXT: '.txt'>]
# digest: 7c2ac02c03563afcf9b574c7e56c153a
# maxvocab: 10000
# maxtokens: 100
# clean_fn: <function clean at 0x0000022C5FD44B80>

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, StripAccents, Sequence, NFD, Replace

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(special_tokens=["<unk>"], vocab_size=10000)
tokenizer.pre_tokenizer = Whitespace()
normalizer = Sequence([NFD(), Replace("<br />", ""), StripAccents(), Lowercase()])
tokenizer.normalizer = normalizer
tokenizer.train_from_iterator(traindataset, trainer=trainer)  # the vocab size is 10000
print(f"the vocab size is {tokenizer.get_vocab_size()}")

# BpeTrainer
print(tokenizer.get_vocab())

from torch.nn.utils.rnn import pad_sequence
import torch

Tensor = torch.Tensor

class Preprocessor:
    def __init__(
        self, max: int, tokenizer
    ) -> None:
        self.max = max
        self.tokenizer = tokenizer

    def cast_label(self, label: str) -> int:
        if label == "neg":
            return 0
        else:
            return 1

    def __call__(self, batch: list) -> tuple[Tensor, Tensor]:
        labels, text = [], []
        for x, y in batch:
            tokens = torch.tensor(self.tokenizer.encode(x).ids)
            tokens = tokens[:self.max]
            text.append(tokens)
            labels.append(self.cast_label(y))

        text_ = pad_sequence(text, batch_first=True, padding_value=0)
        return text_, torch.tensor(labels)

preprocessor = Preprocessor(256, tokenizer)
streamers = imdbdatasetfactory.create_datastreamer(batchsize=32, preprocessor=preprocessor)

train = streamers["train"]
batch = train.batchloop()
print(batch)

train = streamers["train"]
print(f"number of batches {len(train)}")  # number of batches 781
trainstreamer = train.stream()
validstreamer = streamers["valid"].stream()
X, y = next(iter(trainstreamer))
print(X.shape, y.shape)  # (torch.Size([32, 256]), torch.Size([32]))

# Setup accuracy and loss_fn (this is a classification problem with two classes, 0 and 1)
accuracy = metrics.Accuracy()
loss_fn = torch.nn.CrossEntropyLoss()
log_dir = Path("logs/nlp/").resolve()
print(log_dir)  # WindowsPath('C:/Users/avtnl/Documents/HU/ML/MADS-MachineLearning-course/notebooks/5_attention/logs/nlp')

# Basic config. We need to specify the vocabulary lenght for the embedding layer. Trainsteps are set to just 100 batches for speedup in the demo.
from mltrainer import TrainerSettings, ReportTypes

settings = TrainerSettings(
    epochs=3,
    metrics=[accuracy],
    logdir=log_dir,
    train_steps=100,
    valid_steps=25,
    reporttypes=[ReportTypes.TENSORBOARD],
    scheduler_kwargs={"factor": 0.5, "patience": 5},
)
print(settings)

# epochs: 3
# metrics: [Accuracy]
# logdir: C:\Users\avtnl\Documents\HU\ML\MADS-MachineLearning-course\notebooks\5_attention\logs\nlp
# train_steps: 100
# valid_steps: 25
# reporttypes: [<ReportTypes.TENSORBOARD: 'TENSORBOARD'>]
# optimizer_kwargs: {'lr': 0.001, 'weight_decay': 1e-05}
# scheduler_kwargs: {'factor': 0.5, 'patience': 5}
# earlystop_kwargs: {'save': False, 'verbose': True, 'patience': 10}

from mltrainer.rnn_models import ModelConfig
config = ModelConfig(
    input_size=tokenizer.get_vocab_size(),
    hidden_size=128,
    dropout=0.1,
    num_layers=1,
    output_size=2,
)
print(config)  # ModelConfig(input_size=10000, hidden_size=128, num_layers=1, output_size=2, dropout=0.1)

model = NLPmodel(config)
print(model)  # c:\Users\avtnl\Documents\HU\ML2\ml2\Lib\site-packages\torch\nn\modules\rnn.py:123: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.1 and num_layers=1
                       # warnings.warn(

NLPmodel(
  (emb): Embedding(10000, 128)
  (rnn): GRU(128, 128, batch_first=True, dropout=0.1)
  (linear): Linear(in_features=128, out_features=2, bias=True)
)

# The base NLP model is just a GRU, with an embedding as a first layer.

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using MPS")
elif torch.cuda.is_available():
    device = "cuda:0"
    print("using cuda")
else:
    device = "cpu"
    print("using cpu")

optimizer = optim.Adam
scheduler = optim.lr_scheduler.ReduceLROnPlateau

trainer = Trainer(
    model=model,
    settings=settings,
    loss_fn=loss_fn,
    optimizer=optimizer,
    traindataloader=trainstreamer,attentionmodel = AttentionNLP(config)

attentiontrainer = Trainer(
    model=attentionmodel,
    settings=settings,
    loss_fn=loss_fn,
    optimizer=optim.Adam,
    traindataloader=trainstreamer,
    validdataloader=validstreamer,
    scheduler=optim.lr_scheduler.ReduceLROnPlateau,
    device=device,
    )

# Compare the impact of attention
attentiontrainer.loop()
    validdataloader=validstreamer,
    scheduler=scheduler,
    device=device,
    )

trainer.loop()






