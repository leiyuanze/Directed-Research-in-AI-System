import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

def prepare_cifar10_datasets(train_transform=None):

    # Define the transformation: Convert images to tensors and normalize them
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if train_transform is None:
        train_transform = test_transform
    

    # Download and load the CIFAR-10 training and test datasets
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)



    # TODO: Randomly split the **test** dataset into validation and test sets with an **8:2** ratio.
    #       You can use `torch.Subset` or `torch.utils.data.random_split` for this purpose.
    #       Note: In practice, validation datasets are typically split from the training set. 
    #       However, for simplicity in this case, we are splitting from the test set such that there's no need to modify the existing `transform`.

    # raise NotImplementedError()

    total = len(test_dataset)
    valid_size = int(total * 0.8)
    test_size = total - valid_size
    generator = torch.Generator().manual_seed(0)
    valid_dataset, test_dataset = random_split(
        test_dataset, [valid_size, test_size], generator = generator
    )

    return train_dataset, valid_dataset, test_dataset
