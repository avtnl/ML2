# hyper_03.py
import os
from pathlib import Path
import requests
import zipfile
from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory

cudnn.benchmark = True
plt.ion()  # interactive mode


def get_hymenoptera():
    datadir = Path.home() / ".cache/mads_datasets/hymenoptera_data"
    url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"

    if not (datadir / "hymenoptera_data").exists():
        logger.info(f"Creating directory {datadir}")
        datadir.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading hymenoptera_data.zip...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        zip_path = datadir / "hymenoptera_data.zip"
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(datadir)
        zip_path.unlink()
        logger.success("Dataset ready!")
    else:
        logger.info("Dataset already exists.")

    return datadir / "hymenoptera_data"


def create_dataloaders(datadir):
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

    image_datasets = {
        x: datasets.ImageFolder(datadir / x, data_transforms[x])
        for x in ['train', 'val']
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=4,
            shuffle=(x == 'train'),
            num_workers=4,          # works great on Windows when protected
            pin_memory=True,
            persistent_workers=True
        )
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names


# === Visualization helpers ===
def imshow(img, title=None):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def visualize_model(model, dataloaders, class_names, device, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(10, 8))

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size(0)):
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.show()
                    return
                images_so_far += 1

                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f"Pred: {class_names[preds[j]]}\nTrue: {class_names[labels[j]]}")
                imshow(inputs.cpu().data[j])

    model.train(mode=was_training)
    plt.show()


# === Training functions (same as before, slightly cleaned) ===
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = running_corrects = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    return running_loss / len(dataloader.dataset), running_corrects.double() / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = running_corrects = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    return running_loss / len(dataloader.dataset), running_corrects.double() / len(dataloader.dataset)


def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device):
    best_acc = 0.0
    best_path = "best_model.pth"

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, device)
        val_loss, val_acc = validate(model, dataloaders['val'], criterion, device)
        scheduler.step()

        logger.info(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        logger.info(f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            logger.success(f"New best accuracy: {best_acc:.4f}")

    model.load_state_dict(torch.load(best_path))
    logger.info("Training finished - best model loaded.")
    return model


# ========================
#      MAIN GUARD
# ========================
if __name__ == '__main__':
    # --- Dataset ---
    datadir = get_hymenoptera()

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        logger.warning("Running on CPU - slow!")

    # --- DataLoaders ---
    dataloaders, dataset_sizes, class_names = create_dataloaders(datadir)

    # --- Show some training images ---
    inputs, classes = next(iter(dataloaders['train']))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])
    plt.show()

    # --- 1. Fine-tuning full ResNet18 ---
    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    model_ft.fc = nn.Linear(model_ft.fc.in_features, 2)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    logger.info("=== Fine-tuning full ResNet18 ===")
    model_ft = train_model(model_ft, criterion, optimizer, scheduler, num_epochs=15,
                           dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=device)

    visualize_model(model_ft, dataloaders, class_names, device)

    # --- 2. Feature extraction ---
    model_fe = models.resnet18(weights='IMAGENET1K_V1')
    for param in model_fe.parameters():
        param.requires_grad = False
    model_fe.fc = nn.Linear(model_fe.fc.in_features, 2)
    model_fe = model_fe.to(device)

    optimizer_fe = optim.SGD(model_fe.fc.parameters(), lr=0.001, momentum=0.9)
    scheduler_fe = lr_scheduler.StepLR(optimizer_fe, step_size=7, gamma=0.1)

    logger.info("=== Feature extraction mode ===")
    model_fe = train_model(model_fe, criterion, optimizer_fe, scheduler_fe, num_epochs=15,
                           dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=device)

    visualize_model(model_fe, dataloaders, class_names, device)