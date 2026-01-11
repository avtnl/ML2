# hyper_04_flowers_fixed.py
# Flowers102 + ResNet18 transfer learning – fully local, Windows + CUDA

from pathlib import Path
from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# Windows multiprocessing guard
# ------------------------------------------------------------------
if __name__ == "__main__":

    try:
        from mads_datasets import DatasetFactoryProvider, DatasetType
        from mltrainer import Trainer, TrainerSettings, ReportTypes
        from mltrainer.metrics import Accuracy
    except ImportError as e:
        raise ImportError("Run: pip install mads-datasets mltrainer loguru") from e

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    logger.info("Loading Flowers102 dataset...")
    factory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
    streamers = factory.create_datastreamer(batchsize=32, num_workers=4)

    # ------------------------------------------------------------------
    # 2. Tensor-only transforms (NO ToTensor / ToPILImage!)
    # ------------------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ------------------------------------------------------------------
    # 3. Preprocessor that works with tensors returned by mads_datasets
    # ------------------------------------------------------------------
    class TensorOnlyPreprocessor:
        def __init__(self, transform):
            self.transform = transform

        def __call__(self, batch):
            imgs, labels = zip(*batch)                 # list of (tensor, label)
            imgs = torch.stack(imgs)                   # (B, C, H, W) tensor
            imgs = self.transform(imgs)                # apply tensor-only transforms
            labels = torch.tensor(labels, dtype=torch.long)
            return imgs, labels

    # Apply correct preprocessors
    streamers["train"].preprocessor = TensorOnlyPreprocessor(train_transform)
    streamers["valid"].preprocessor = TensorOnlyPreprocessor(val_transform)

    train_stream = streamers["train"].stream()
    valid_stream = streamers["valid"].stream()

    # ------------------------------------------------------------------
    # 4. Test pipeline
    # ------------------------------------------------------------------
    logger.info("Testing data pipeline...")
    X, y = next(train_stream)
    logger.success(f"Success! X.shape: {X.shape}, y.shape: {y.shape}")

    # Visualize batch
    def show_batch(tensor):
        img = tensor[:9].clone()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img = img * std + mean
        img = img.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()

        plt.figure(figsize=(10,10))
        for i in range(9):
            plt.subplot(3, 3, i+1)
            plt.imshow(img[i])
            plt.axis('off')
        plt.suptitle("Flowers102 – Training batch")
        plt.show()

    show_batch(X)

    # ------------------------------------------------------------------
    # 5. Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.success(f"Using {device} → {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")

    # ------------------------------------------------------------------
    # 6. Model
    # ------------------------------------------------------------------
    logger.info("Loading pretrained ResNet18...")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, 5)
    model = model.to(device)

    # Quick forward test
    out = model(X.to(device))
    logger.info(f"Model output: {out.shape}")

    # ------------------------------------------------------------------
    # 7. Training settings
    # ------------------------------------------------------------------
    settings = TrainerSettings(
        epochs=30,
        metrics=[Accuracy()],
        logdir="modellogs/flowers",
        train_steps=len(streamers["train"]),
        valid_steps=len(streamers["valid"]),
        reporttypes=[ReportTypes.TENSORBOARD],
        optimizer_kwargs={"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4},
        scheduler_kwargs={"step_size": 10, "gamma": 0.5},
        earlystop_kwargs=None,
        batchsize=32,
        device=device,
    )

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optim.SGD,
        traindataloader=train_stream,
        validdataloader=valid_stream,
        scheduler=lr_scheduler.StepLR,
        device=device,
    )

    # ------------------------------------------------------------------
    # 8. TRAIN!
    # ------------------------------------------------------------------
    logger.info("Starting training – ~60-90 seconds per epoch on RTX 5070")
    trainer.loop()

    # ------------------------------------------------------------------
    # 9. Save model
    # ------------------------------------------------------------------
    path = Path("flowers102_resnet18_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    }, path)
    logger.success(f"Done! Model saved → {path}")

    print("\nOpen TensorBoard:")
    print("tensorboard --logdir modellogs/flowers")