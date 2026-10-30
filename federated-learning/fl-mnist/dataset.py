import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def load_data(client_id, num_clients=3):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transform
    )

    # Divide dataset among clients
    total_size = len(dataset)
    client_size = total_size // num_clients

    lengths = [client_size] * num_clients

    # Add remaining samples to last client
    lengths[-1] += total_size - sum(lengths)

    datasets_split = random_split(
        dataset,
        lengths,
        generator=torch.Generator().manual_seed(42)
    )

    client_dataset = datasets_split[client_id]

    train_size = int(0.8 * len(client_dataset))
    test_size = len(client_dataset) - train_size

    train_dataset, test_dataset = random_split(
        client_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32
    )

    return train_loader, test_loader