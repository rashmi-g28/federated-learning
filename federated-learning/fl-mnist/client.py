import sys

import torch
import flwr as fl

from model import Net
from dataset import load_data


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def train(model, train_loader):

    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01
    )

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1):

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()


def test(model, test_loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = correct / total

    return accuracy


class FlowerClient(
    fl.client.NumPyClient
):

    def __init__(self, client_id):

        self.model = Net().to(DEVICE)

        self.train_loader, self.test_loader = load_data(
            client_id
        )

    def get_parameters(self, config):

        return [
            value.cpu().numpy()
            for value in self.model.state_dict().values()
        ]

    def fit(self, parameters, config):

        self.model.load_state_dict(
            dict(
                zip(
                    self.model.state_dict().keys(),
                    [
                        torch.tensor(p)
                        for p in parameters
                    ]
                )
            )
        )

        train(
            self.model,
            self.train_loader
        )

        return self.get_parameters(config), len(
            self.train_loader.dataset
        ), {}

    def evaluate(self, parameters, config):

        self.model.load_state_dict(
            dict(
                zip(
                    self.model.state_dict().keys(),
                    [
                        torch.tensor(p)
                        for p in parameters
                    ]
                )
            )
        )

        accuracy = test(
            self.model,
            self.test_loader
        )

        return 0.0, len(
            self.test_loader.dataset
        ), {"accuracy": accuracy}


if __name__ == "__main__":

    client_id = int(sys.argv[1])

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(client_id).to_client()
    )