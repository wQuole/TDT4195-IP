import torch
import torchvision
import numpy as np


def load_data(batch_size_train, batch_size_test):
    train_loader = torch.utils.data.Dataloader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.Dataloader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ])),
        batch_size=batch_size_test, shuffle=True)


def main():
    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1024
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = np.random.seed(1)
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    load_data(batch_size_train, batch_size_test)



if __name__ == '__main__':
    main()