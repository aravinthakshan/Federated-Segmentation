import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np

def get_dataset(seed=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    return trainset, testset

def iid_split(dataset, num_clients, seed=None):
    if seed is not None:
        np.random.seed(seed)
    num_items = int(len(dataset) / num_clients)
    all_idxs = np.arange(len(dataset))
    np.random.shuffle(all_idxs)

    client_idxs = [all_idxs[i*num_items:(i+1)*num_items] for i in range(num_clients)]
    return client_idxs

def non_iid_split(dataset, num_clients, num_shards=200, shards_per_client=2, seed=None):
    """
    Non-IID partition by sorting dataset by label and splitting into shards
    """
    if seed is not None:
        np.random.seed(seed)

    # get labels
    targets = np.array(dataset.targets)
    idxs = np.arange(len(dataset))

    # sort by label
    idxs_labels = np.vstack((idxs, targets))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    shards_size = len(dataset) // num_shards
    shards = [idxs[i*shards_size:(i+1)*shards_size] for i in range(num_shards)]
    np.random.shuffle(shards)

    client_idxs = []
    for i in range(num_clients):
        shards_for_client = shards[i*shards_per_client:(i+1)*shards_per_client]
        client_idxs.append(np.hstack(shards_for_client))

    return client_idxs

def get_client_loaders(dataset, client_idxs, batch_size):
    client_loaders = []
    for idxs in client_idxs:
        subset = Subset(dataset, idxs)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
    return client_loaders

def get_test_loader(testset, batch_size):
    return DataLoader(testset, batch_size=batch_size, shuffle=False)
