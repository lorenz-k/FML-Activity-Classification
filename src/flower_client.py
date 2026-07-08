import argparse
import os

import flwr as fl
import torch
from torch import nn

from src.data_loader import get_client_loader
from src.flower_utils import get_model_parameters, set_model_parameters
from src.local_model import ActivityMLP
from src.local_training import get_device, train_one_epoch


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


class HarFlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        hidden_dim: int,
        dropout: float,
        data_dir=None,
    ):
        self.client_id = client_id
        self.epochs = epochs
        self.device = get_device()
        self.train_loader = get_client_loader(
            client_id, batch_size=batch_size, data_dir=data_dir
        )
        self.model = ActivityMLP(hidden_dim=hidden_dim, dropout=dropout).to(self.device)    # new random init
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)    # loads param arrays into the state dict

        round_epochs = int(config.get("local_epochs", self.epochs))
        train_loss = 0.0
        train_accuracy = 0.0
        for _ in range(round_epochs):
            train_loss, train_accuracy = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
            )

        num_examples = len(self.train_loader.dataset)
        return (
            get_model_parameters(self.model),
            num_examples,
            {
                "client_id": self.client_id,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
            },
        )


def main():
    client_id_env = os.environ.get("CLIENT_ID")
    parser = argparse.ArgumentParser(description="Start one Flower HAR client.")
    parser.add_argument(
        "--client-id",
        type=int,
        default=int(client_id_env) if client_id_env is not None else None,
        required=client_id_env is None,
        help="Client shard id.",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=env_str("SERVER_ADDRESS", "127.0.0.1:8080"),     # 127.0.0.1 calls stay on local machines, adress to server's port
        help="Flower server address.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=env_int("LOCAL_EPOCHS", 1),
        help="Local epochs per round.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=env_int("BATCH_SIZE", 64),
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=env_float("LR", 1e-3),
        help="Learning rate.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=env_int("HIDDEN_DIM", 128),
        help="Hidden layer size.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=env_float("DROPOUT", 0.2),
        help="Dropout probability.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=env_str("FML_DATA_DIR", None),
        help="Directory holding client_<id>.npz shards (default: outputs/har).",
    )
    args = parser.parse_args()

    client = HarFlowerClient(
        client_id=args.client_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        data_dir=args.data_dir,
    )

    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
