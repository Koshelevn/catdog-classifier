from pathlib import Path

import torchvision
from torchvision import transforms


def get_datasets(
    data_path, train, val, test, size_height, size_width, image_mean, image_std
):

    data = Path(data_path)

    transformer = transforms.Compose(
        [
            transforms.Resize((size_height, size_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                image_mean, image_std
            ),  # normalize image data per-channel
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        str(data / train), transform=transformer
    )
    val_dataset = torchvision.datasets.ImageFolder(str(data / val), transform=transformer)
    test_dataset = torchvision.datasets.ImageFolder(
        str(data / test), transform=transformer
    )

    return train_dataset, val_dataset, test_dataset
