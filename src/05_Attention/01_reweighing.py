from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
trainer = WordLevelTrainer(special_tokens=["[UNK]"])
tokenizer.pre_tokenizer = Whitespace() # This pre-tokenizer simply splits using the following regex: \w+|[^\w\s]+

corpus = ["ik zit op de bank", "ik werk bij de bank", "bij de bank is het heel druk"]

tokenizer.train_from_iterator(corpus, trainer=trainer)
print(tokenizer.get_vocab())

x = tokenizer.encode(corpus[0])
print(x.ids)  # [4, 11, 9, 2, 1]

import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
vocab_len = tokenizer.get_vocab_size()
emb = nn.Embedding(num_embeddings=vocab_len, embedding_dim=4, padding_idx=0)
tokenized_corpus = pad_sequence([torch.tensor(tokenizer.encode(s).ids) for s in corpus], batch_first=True)
embeddings = emb(tokenized_corpus)
print(embeddings, embeddings.shape)

bank1 = embeddings[0][4]
bank2 = embeddings[1][4]
print(bank1, bank2)

# Attention: Now we will start with the attention mechanism. We need a key, query and value. Because we use self attention, these are just clones.
key = embeddings.detach().clone()
query = embeddings.detach().clone()
values = embeddings.detach().clone()
print(key.shape)  # torch.Size([3, 7, 4])

d_features = torch.tensor(query.shape[-1])
print(d_features)  # tensor(4)

dots = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(d_features)
print(dots.shape)  # torch.Size([3, 7, 7])

weights = nn.Softmax(dim=-1)(dots)
print(weights[0])

activations = torch.bmm(weights, values)
print(activations.shape, embeddings.shape)  # (torch.Size([3, 7, 4]), torch.Size([3, 7, 4]))

bank1 = activations[0][4]
bank2 = activations[1][4]
print(bank1, bank2)

# torch has a multihead attention implemented. With that, we can add a mask to cover the padding.
mask = tokenized_corpus == 0
multihead = nn.MultiheadAttention(embed_dim=4, num_heads=2, batch_first=True)
attn, attn_w = multihead(query, key, values, key_padding_mask=mask)
print(attn.shape)  # torch.Size([3, 7, 4])

# It is possible to visualize the weights.
import seaborn as sns
plt.figure(figsize=(10, 10))
labels = corpus[0].split()
labels = labels + ["PAD", "PAD"]
plot = sns.heatmap(attn_w[0].detach().numpy())
plot.set_xticklabels(labels);
plot.set_yticklabels(labels);