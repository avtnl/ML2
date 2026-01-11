# Original Notebook | NOT WORKING ON LAPTOP!!!
import os
# Check if the notebook is running on Google Colab
colab = False
if 'COLAB_GPU' in os.environ:
    colab = True
    # Running on Google Colab; install loguru
    !pip install mads_datasets mltrainer loguru
else:
    # Not running on Google Colab; you might be on a local setup
    print("Not running on Google Colab. Ensure dependencies are installed as needed.")

from pathlib import Path
from loguru import logger
import torch
import matplotlib.pyplot as plt

from mads_datasets import DatasetFactoryProvider, DatasetType

flowersfactory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
streamers = flowersfactory.create_datastreamer(batchsize=32)

# === Data augmentation ===
from torchvision import transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

flowersfactory.settings

flowersfactory.settings.img_size = (500, 500)

# === Create dataset ===
if not colab:
    datasets = flowersfactory.create_dataset()
    traindataset = datasets["train"]
    img, lab = traindataset[0]
    logger.info(f"original shape: {img.shape}")
    # original shape: torch.Size([3, 500, 500])
    transformed_img = data_transforms["train"](img)
    logger.info(f"transformed shape: {transformed_img.shape}")
    # transformed shape: torch.Size([3, 224, 224])

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img.numpy().transpose(1, 2, 0))
    ax[1].imshow(transformed_img.numpy().transpose(1, 2, 0))

# Instead of using the BasePreprocessor, we will squeeze in the transformer. Lets make that:
class AugmentPreprocessor():
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        X, y = zip(*batch)
        X = [self.transform(x) for x in X]
        return torch.stack(X), torch.stack(y)

# Now we can create an separate preprocessor for train and validation:
trainprocessor= AugmentPreprocessor(data_transforms["train"])
validprocessor = AugmentPreprocessor(data_transforms["val"])

# And add that as the preprocessor for train and validation streamers. We do it like this because by default we can only provide a single preprocessor for both training and validation.
train = streamers["train"]
valid = streamers["valid"]
train.preprocessor = trainprocessor
valid.preprocessor = validprocessor
trainstreamer = train.stream()
validstreamer = valid.stream()

# Check if this works
X, y = next(trainstreamer)
print("X.shape, y.shape:", X_shape, y_shape)

# And lets visualise a random batch of images
import matplotlib.pyplot as plt
import numpy as np
img = X.permute(0, 2, 3, 1).numpy()
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img = std * img + mean
img = np.clip(img, 0, 1)
fig, axs = plt.subplots(3, 3, figsize=(10,10))
axs = axs.ravel()
for i in range(9):
    axs[i].imshow(img[i])

# Instead of building our own resnet, we will just download a pretrained version. This saves us many hours of training.
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)

ResNet18_Weights.DEFAULT

yhat = resnet(X)
print("yhat.shape:", yhat_shape)

# However, the resnet is trained for 1000 classes. We have just 5... so we will swap the last layer and retrain the model.

#First, we freeze all pretrained layers:
for name, param in resnet.named_parameters():
    param.requires_grad = False

# If you study the resnet implementation on [github](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L206) you can see that the last layer is named `.fc`, like this:
# self.fc = nn.Linear(512 * block.expansion, num_classes)
# This is a Linear layer, mapping from 512 * block.expansion to num_classes. So we will swap that for our own. To do so we need to figure out how many features go into the .fc layer.
# We can retrieve the incoming amount of features for the current `.fc` with `.in_features`
print(type(resnet.fc))
in_features = resnet.fc.in_features
in_features

# Let's swap that layer with a minimal network. Sometimes just a linear layer is enough, sometimes you want to add two layers and some dropout. Play around to see the difference!
import torch.nn as nn

resnet.fc = nn.Sequential(
    nn.Linear(in_features, 5)
    # nn.Linear(in_features, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 5)
)

yhat = resnet(X)
print("yhat.shape:", yhat_shape)

# So, we have a fully trained resnet, but we added two layers at the end that transforms everything into 5 classes. These layers are random, so we need to train them for some epochs
from mltrainer import metrics
accuracy = metrics.Accuracy()

# This will take some time to train (about 4 min per epoch), you could scale down to amount of trainsteps to speed things up.
# You will start with a fairly high learning rate (0.01), and if the learning stops, after patience epochs the learning rate gets halved.
len(train), len(valid)

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"
    logger.warning("This model will take 15-20 minutes on CPU. Consider using accelaration, eg with google colab (see button on top of the page)")
logger.info(f"Using {device}")

# We are going to use SGD as optimizer and a stepLR as scheduler.
from torch import optim
optimizer = optim.SGD
scheduler = optim.lr_scheduler.StepLR

# To make this actually learn enough, you should increase the epochs to about 30.
from mltrainer import Trainer, TrainerSettings, ReportTypes

settings = TrainerSettings(
    epochs=3,
    metrics=[accuracy],
    logdir="modellogs/flowers",
    train_steps=len(train),
    valid_steps=len(valid),
    reporttypes=[ReportTypes.TENSORBOARD],
    optimizer_kwargs= {'lr': 0.1, 'weight_decay': 1e-05, 'momentum': 0.9},
    scheduler_kwargs= {'step_size' : 10, 'gamma' : 0.1},
    earlystop_kwargs= None,
)
settings

trainer = Trainer(
    model=resnet,
    settings=settings,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    traindataloader=trainstreamer,
    validdataloader=validstreamer,
    scheduler=scheduler,
    device=device,
    )

trainer.loop()

