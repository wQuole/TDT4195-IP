# It should not be required to change this code
import torchvision
import torch


def load_dataset(batch_size,
                 image_transform,
                 root_dir="data/"):

    dataset_train = torchvision.datasets.MNIST(
        root=root_dir,
        download=True,
        transform=image_transform
    )
    dataset_test = torchvision.datasets.MNIST(
        root=root_dir,
        download=True,
        train=False,
        transform=image_transform
    )
    dataloader_train = torch.utilz.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
    )
    dataloader_test = torch.utilz.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False
    )
    return dataloader_train, dataloader_test
