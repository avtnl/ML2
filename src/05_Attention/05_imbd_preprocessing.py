from pathlib import Path
import tokenizers as tk
import mltrainer
print(mltrainer.__version__)   # '0.2.5'

# Load the IMDB dataset (the MNIST for language models)
from mads_datasets import DatasetFactoryProvider, DatasetType
imdbdatasetfactory = DatasetFactoryProvider.create_factory(DatasetType.IMDB)
datasets = imdbdatasetfactory.create_dataset()

from mads_datasets import DatasetFactoryProvider, DatasetType
imdbdatasetfactory = DatasetFactoryProvider.create_factory(DatasetType.IMDB)
datasets = imdbdatasetfactory.create_dataset()

traindataset = datasets["train"]
testdataset = datasets["valid"]

# print first datapoint from about 50K movie reviews
x, y = traindataset[10]
print(x, y) # ('This interesting Giallo boosts a typical but still thrilling plot and a really sadistic killer that obviously likes to hunt his victims down before murdering them in gory ways.<br /><br />Directed by Emilio P. Miraglia who, one year earlier, also made the very interesting "La Notte che Evelyn Usci della Tomba" (see also my comment on that one), the film starts off a little slow, but all in all, no time is wasted with unnecessary sub plots or sequences.<br /><br />This film is a German-Italian coproduction, but it was released in Germany on video only in a version trimmed by 15 minutes of plot under the stupid title "Horror House". At least the murder scenes, which will satisfy every gorehound, are fully intact, and the viewer still gets the killer\'s motive at the end. But the Italian version containing all the footage is still the one to look for, of course.<br /><br />A convincing Giallo with obligatory twists and red herrings, "La Dama Rossa Uccide Sette Volte" is highly recommended to Giallo fans and slightly superior to Miraglia\'s above mentioned other thriller.',
 'pos')

for x,y in traindataset:
    if "<br" in x:
        print(x)
        break

# test = "I'd like to think myself as a fairly open minded guy and it takes a lot(!) for me to dislike a movie but this one is without a doubt one of the suckiest, crappiest movie I've ever seen!<br />From the decrepit ranks of the already over-saturated \'Hillybilly Horror\'"

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, StripAccents, Sequence, NFD, Replace

normalizer = Sequence([NFD(), Replace("<br />", ""), StripAccents(), Lowercase()])
normalizer.normalize_str(test)

# "i'd like to think myself as a fairly open minded guy and it takes a lot(!) for me to dislike a movie but this one is without a doubt one of the suckiest, crappiest movie i've ever seen!from the decrepit ranks of the already over-saturated 'hillybilly horror'"

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(special_tokens=["<unk>"])
tokenizer.pre_tokenizer = Whitespace()
tokenizer.normalizer = normalizer
tokenizer.train_from_iterator(traindataset, trainer=trainer)
print(f"the vocab size is {tokenizer.get_vocab_size()}")  # the vocab size is 30000

print(tokenizer.get_vocab())  # This maps a sentence of words to a sequence of integers.

import torch
print(torch.tensor(tokenizer.encode(test).ids))  # … and we create a tensor with this

from typing import List, Tuple, Optional, Callable
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

    def __call__(self, batch: List) -> Tuple[Tensor, Tensor]:
        labels, text = [], []
        for x, y in batch:
            tokens = torch.tensor(self.tokenizer.encode(x).ids)
            tokens = tokens[:self.max]
            text.append(tokens)
            labels.append(self.cast_label(y))

        text_ = pad_sequence(text, batch_first=True, padding_value=0)
        return text_, torch.tensor(labels)

corpus = [("The cat sat on the mat", "neg"), ("Where is the cat?", "pos"), ("The cat is blasé", "neg")]
pp = Preprocessor(max = 5, tokenizer=tokenizer)
print(pp(corpus))

# (tensor([[ 125, 1723, 1149,  130,  125],
#               [ 441,  128,  125, 1723,   33],
#               [ 125, 1723,  128,  394,  560]]),
# tensor([0, 1, 0]))

# Preprocessing is necessary to:
# - cut of long sentences to get equal length. 100 words will be enough to get the sentiment in most cases
# - we need to cast the labels "neg" and "pos" to integers
# - we also pad if a sentence is shorter than the max lenght

# We can feed the preprocessor to the default dataloader from torch
from torch.utils.data import DataLoader

preprocessor = Preprocessor(max=512, tokenizer=tokenizer)
dataloader = DataLoader(
    traindataset, collate_fn=preprocessor, batch_size=32, shuffle=True
)

# Get batched sentences and labels
x, y = next(iter(dataloader))
print(x.shape, y.shape)  # (torch.Size([32, 512]), torch.Size([32]))
print(x[0])

# Note: All this code is wrapped into the DatasetFactoryProvider, which you can see in the next notebook.
