# Original Notebook | NOT WORKING ON LAPTOP!!!
import os

# Check if the notebook is running on Google Colab
if 'COLAB_GPU' in os.environ:
    # Running on Google Colab; install loguru
    !pip install loguru
else:
    # Not running on Google Colab; you might be on a local setup
    print("Not running on Google Colab. Make sure you have hardware accelaration available.")

from pathlib import Path
import requests
import zipfile
from loguru import logger

def get_hymenoptera():
    datadir = Path.home() / ".cache/mads_datasets/hymenoptera_data"
    url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
    if not datadir.exists():
        logger.info(f"Creating directory {datadir}")
        datadir.mkdir(parents=True)

        response = requests.get(url)
        zip_file_path = datadir / "hymenoptera_data.zip"
        with open(zip_file_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Extracting {zip_file_path}")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(datadir)
        zip_file_path.unlink()
    else:
        logger.info(f"Directory {datadir} already exists, skipping download.")
    return datadir / "hymenoptera_data"

datadir = get_hymenoptera()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True
plt.ion()   # interactive mode

# === Load Data ===
def create_dataloaders():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(datadir / x, data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names

dataloaders, dataset_sizes, class_names = create_dataloaders()

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"
    logger.warning("This model will take 15-20 minutes on CPU. Consider using accelaration, eg with google colab (see button on top of the page)")
logger.info(f"Using {device}")

# === Visualize a few images ===
def imshow(img, title):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
img = torchvision.utils.make_grid(inputs)

imshow(img, title=[class_names[x] for x in classes])

# === Train the model ===
def train(model, dataloader, lossfn, optimizer, device):
    model.train()
    train_loss: float = 0.0
    train_acc: float = 0.0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        yhat = model(x)
        loss = lossfn(yhat, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, acc = torch.max(yhat, 1)
        train_acc += torch.sum(acc == y.data)
    return train_loss, train_acc

def test(model, dataloader, lossfn, optimizer, scheduler, device):
    model.eval()
    test_loss: float = 0.0
    test_acc: float = 0.0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        yhat = model(x)
        loss = lossfn(yhat, y)
        test_loss += loss.item()
        _, acc = torch.max(yhat, 1)
        test_acc += torch.sum(acc == y.data)
    scheduler.step(test_loss)
    return test_loss, test_acc

def train_model(model, lossfn, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device):
    with TemporaryDirectory() as tempdir:
        best_model_params_path = Path(tempdir) / 'best_model_params.pt'
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0 # we will only save the best model

        for epoch in range(num_epochs):
            logger.info(f'Epoch {epoch}/{num_epochs - 1}')
            train_loss, train_acc = train(model, dataloaders['train'], lossfn, optimizer, device)
            train_loss = train_loss / dataset_sizes['train']
            train_acc = train_acc / dataset_sizes['train']
            logger.info(f'Train Loss: {train_loss:.4f} Accuracy: {train_acc:.4f}')
            test_loss, test_acc = test(model, dataloaders['val'], lossfn, optimizer, scheduler, device)
            test_loss = test_loss / dataset_sizes['val']
            test_acc = test_acc / dataset_sizes['val']
            logger.info(f'Test Loss: {test_loss:.4f} Accuracy: {test_acc:.4f}')
            if test_acc > best_acc:
                    best_acc = test_acc
                    logger.info(f"New best accuracy: {best_acc:.4f}, saving model")
                    torch.save(model.state_dict(), best_model_params_path)
        model.load_state_dict(torch.load(best_model_params_path))
        return model

# === Visualizing the model predictions ===
def visualize_model(model, num_images=6):
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                title = class_names[preds[j]]
                ax.set_title(f'predicted: {title}')
                imshow(inputs.cpu().data[j], title)

                if images_so_far == num_images:
                    return

# === Finetuning the ConvNet ===
model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
model.fc = nn.Linear(num_ftrs, 2)

model = model.to(device)
lossfn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs

settings = {'step_size' : 7, 'gamma' : 0.1}
scheduler = lr_scheduler.StepLR(optimizer, **settings)

epochs = 3
model_conv = train_model(
    model = model,
    lossfn = lossfn,
    optimizer = optimizer,
    scheduler = scheduler,
    num_epochs = epochs,
    dataloaders = dataloaders,
    dataset_sizes = dataset_sizes,
    device = device,
)

# === Train and evaluate ===
visualize_model(model)

# === ConvNet a fixed feature extractor ===
model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

lossfn = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# === Train and evaluate ===
epochs = 3
model_conv = train_model(
    model = model_conv,
    lossfn = lossfn,
    optimizer = optimizer,
    scheduler = scheduler,
    num_epochs = epochs,
    dataloaders = dataloaders,
    dataset_sizes = dataset_sizes,
    device = device,
)

visualize_model(model_conv)
