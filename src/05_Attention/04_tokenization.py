import tokenizers as tk
import torch
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, StripAccents, Sequence

corpus = ["The cat sat on the mat", "Where is the cat?", "The cat is blasé"]

from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
normalizer = normalizers.Sequence([NFD(), StripAccents()])
normalizer.normalize_str(corpus[2])

# The NFD (Normalization Form D) normalizer tokenizer performs Unicode normalization on strings, specifically converting them into the Canonical Decomposition form.
# Tokenizers use NFD normalization to ensure consistency in token representation, which is crucial for training effective machine learning models:
# - Standardization: It guarantees that different ways of decomposing the same character 
# - Vocabulary Reduction: By decomposing characters, it can simplify the model's vocabulary. Instead of having separate tokens for 'a', 'aˊ', 'aˋ', etc., the tokenizer might be able to create one token for the base 'a' and separate tokens for the combining accents, leading to a smaller, more efficient vocabulary.

tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
trainer = WordLevelTrainer(special_tokens=["[UNK]"])
tokenizer.pre_tokenizer = Whitespace() # This pre-tokenizer simply splits using the following regex: \w+|[^\w\s]+
normalizer = Sequence([NFD(), StripAccents(), Lowercase()])
# Seqeunce allows concatenating multiple other Normalizer as a Sequence. All the normalizers run in sequence in the given order.
tokenizer.normalizer = normalizer
tokenizer.train_from_iterator(corpus, trainer=trainer)
print(tokenizer.get_vocab())

enc = tokenizer.encode("cat")
print(enc.ids)  # [2]

enc = tokenizer.encode("The cat is drinking")
print(enc.ids)  # [1, 2, 3, 0]

print(tokenizer.decode([0], skip_special_tokens=False))  # 'the cat is [UNK]'

tokenized_sentence = tokenizer.encode(corpus[0])
print(tokenized_sentence.ids)  # [1, 2, 8, 7, 1, 6]

# Original sentence
print(tokenizer.get_vocab())

import torch.nn.functional as F

tokenized_tensor = torch.tensor(tokenized_sentence.ids)
oh = F.one_hot(tokenized_tensor)
print(oh)

vocab_size = tokenizer.get_vocab_size()
print(f"the vocabulary size is {vocab_size}")
hidden_dim = 4

# While this might seem like a nice workaround, it is very memory inefficient. Vocabularies can easily grow into the 10.000+ words!
# So, let's make a more dense space. We simply decide on a dimensionality, and start with assigning a random vector to every word.
embedding = torch.nn.Embedding(
    num_embeddings=vocab_size, embedding_dim=hidden_dim, padding_idx=-2
)
x = embedding(tokenized_tensor)
print(x)

# Reflection
# - we started with a sentence of strings.
# - we map the strings to arbitrary integers
# - the integers are used with an Embedding layer; this is nothing more than a lookup table where every word get's a random vector assigned
# We started with a 6-word sentence. But we ended with a (6, 4) matrix of numbers.
# So, let's say we have a batch of 32 sentences. We can now store this for example as a (32, 15, 6) matrix: batchsize 32, length of every sentence is 15 (use padding if the sentence is smaller), and every word in the sentence represented with 6 numbers.
# This is exactly the same as what we did before with timeseries! We have 3 dimensional tensors, (batch x sequence_length x dimensionality) that we can feed into an RNN!

x_ = x[None, ...]
rnn = torch.nn.GRU(input_size=hidden_dim, hidden_size=16, num_layers=1)

out, hidden = rnn(x_)
print(out.shape, hidden.shape)  # (torch.Size([1, 6, 16]), torch.Size([1, 6, 16]))

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = Whitespace()
normalizer = Sequence([NFD(), StripAccents(), Lowercase()])
# Seqeunce allows concatenating multiple other Normalizer as a Sequence. All the normalizers run in sequence in the given order.
tokenizer.normalizer = normalizer
tokenizer.train_from_iterator(corpus, trainer=trainer)
print((f"the vocabulary size is {tokenizer.get_vocab_size()}"))  # the vocabulary size is 35
print(tokenizer.get_vocab())

enc = tokenizer.encode("The cat is drinking")
print(enc.ids)  # [22, 23, 24, 0, 16, 11, 14, 0, 11, 14, 0]

print(tokenizer.decode(enc.ids, skip_special_tokens=False))  # 'the cat is [UNK] r i n [UNK] i n [UNK]'

def buildBPE(corpus: list[str], vocab_size: int) -> tk.Tokenizer:
    tokenizer = tk.Tokenizer(tk.models.BPE())
    trainer = tk.trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
    )

    # handle spaces better by removing the prefix space
    tokenizer.pre_tokenizer = tk.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = tk.decoders.ByteLevel()

    # train the BPE model  
    tokenizer.train_from_iterator(corpus, trainer)
    tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
    return tokenizer

bpe_tokenizer = buildBPE(corpus, vocab_size=30)
print((f"the vocabulary size is {bpe_tokenizer.get_vocab_size()}"))#  the vocabulary size is 30
bpe_tokenizer.get_vocab()

enc = bpe_tokenizer.encode("The cat is drinking")
print(enc.ids)  # [28, 27, 23, 29, 23, 18, 13, 16, 13, 16]

print(bpe_tokenizer.decode(enc.ids, skip_special_tokens=False))  # The cat is rinin



