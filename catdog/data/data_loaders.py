import os
from pathlib import Path

import torchvision
from torchvision import transforms


DATA_PATH = Path("catdog/data")

# Image size: even though image sizes are bigger than 64, we use this to speed up training
SIZE_H = SIZE_W = 96

# Images mean and std channelwise
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]

transformer = transforms.Compose(
    [
        transforms.Resize((SIZE_H, SIZE_W)),  # scaling images to fixed size
        transforms.ToTensor(),  # converting to tensors
        transforms.Normalize(image_mean, image_std),  # normalize image data per-channel
    ]
)

print(os.curdir)
train_dataset = torchvision.datasets.ImageFolder(
    str(DATA_PATH / "train_11k"), transform=transformer
)
val_dataset = torchvision.datasets.ImageFolder(
    str(DATA_PATH / "val"), transform=transformer
)
test_dataset = torchvision.datasets.ImageFolder(
    str(DATA_PATH / "test_labeled"), transform=transformer
)
