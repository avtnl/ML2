# Dataset
# We will explore this dataset: https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State#

# All data is from one continuous EEG measurement with the Emotiv EEG Neuroheadset. The duration of the measurement was 117 seconds. The eye state was detected via a camera during the EEG measurement and added later manually to the file after analysing the video frames. '1' indicates the eye-closed and '0' the eye-open state.
# All values are in chronological order with the first measured value at the top of the data.
from mads_datasets import datatools
from pathlib import Path
data_dir = Path.home() / ".cache/mads_datasets/egg"
if not data_dir.exists():
    data_dir.mkdir(parents=True)

filename = "EGG.arff"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
datatools.get_file(data_dir=data_dir, filename=filename, url=url, unzip=False)
datapath = data_dir / filename

# Load the arff file with scipy
from scipy.io import arff
data = arff.loadarff(datapath)

# The data is a tuple of a description and observations
print(len(data), type(data))  # (2, tuple)

# Description
print(data[1])

# Dataset: EEG_DATA
# 	AF3's type is numeric
# 	F7's type is numeric
#	F3's type is numeric
#	FC5's type is numeric
#	T7's type is numeric
#	P7's type is numeric
#	O1's type is numeric
#	O2's type is numeric
#	P8's type is numeric
#	T8's type is numeric
#	FC6's type is numeric
#	F4's type is numeric
#	F8's type is numeric
#	AF4's type is numeric
#	eyeDetection's type is nominal, range is ('0', '1')

# There are about 15k observations
print(len(data[0]))  # 14980

# The observations are tuples of floats and a byte as label
print(data[0][0])  # np.void((4329.23, 4009.23, 4289.23, 4148.21, 4350.26, 4586.15, 4096.92, 4641.03, 4222.05, 4238.46, 4211.28, 4280.51, 4635.9, 4393.85, b'0'), dtype=[('AF3', '<f8'), ('F7', '<f8'), ('F3', '<f8'), ('FC5', '<f8'), ('T7', '<f8'), ('P7', '<f8'), ('O1', '<f8'), ('O2', '<f8'), ('P8', '<f8'), ('T8', '<f8'), ('FC6', '<f8'), ('F4', '<f8'), ('F8', '<f8'), ('AF4', '<f8'),
                             # ('eyeDetection', 'S1')])

for x in data[0][0]:
    print(type(x))

# Let's cast the byte ot int
labels = []
for x in data[0]:
    labels.append(int(x[14]))

import numpy as np
print(np.array(labels).mean())  # np.float64(0.4487983978638184)

# About 45% of the data has closed eyes.

# Excercises 1
# - download the data to a given path. You can use the datatools.py method get_file for that, and wrap it with the prerpocessing.
# - build a custom Dataset that yields a $X, y$ tuple of tensors. $X$ should be sequential in time. Remember: a dataset should implement `__get_item__` and `__len__`.
# - You can try to implement your own datafactory. Study all the examples in `mads_datasets` sourcecode.
# - note that you could model this as both a classification task, but also as a sequence-to-sequence task! For this excercise, make it a classification task with consecutive 0s or 1s only.
# - Note that, for a training task, a seq2seq model will probably be more realistic. However, the classification is a nice excercise because it is harder to set up.
# - figure out what the length distribution is of your dataset: how many timestamps do you have for every consecutive sequence of 0s and 1s? On average, median, min, max?
# - create a dataloader that yields timeseries with (batch, sequence_lenght). You can implement: windowed, padded and batched.
#     1. yielding a windowed item should be the easy level
#     2. yielding windowed and padded is medium level 
#     3. yielding windowed, padded and batched is expert level, because the windowing will cause the timeseries to have different sizes. You will need to buffer before you can yield a batch.

# - check if your dataloader works:
#     - it should not give errors because it runs out of data! Either let is stop by itself, or run forever.
#     - batchsize should be consistent (in case 1 and 2, batchsize is 1)
#     - sequence length is allowed to vary

# # Excercise 2
# - build a Dataset that yields sequences of X, y. This time, y is a sequence and can contain both 0s and 1s
# - create a Dataloader with this
# - Test appropriate architectures (RNN, Attention)
# - for the loss, note that you will need a BCELoss instead of a CrossEntroyLoss