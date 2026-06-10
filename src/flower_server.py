import argparse
import os
from pathlib import Path

import flwr as fl
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from torch import nn

from src.data_loader import get_test_loader
from src.flower_utils import (
    get_model_parameters,
    save_model_parameters,
    set_model_parameters,
)
from src.local_model import ActivityMLP
from src.local_training import evaluate, get_device, macro_f1_from_confusion
from src.run_logging import RUNS_DIR, RunLogger, default_run_id


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    train_loss = sum(
        num_examples * float(metric["train_loss"])
        for num_examples, metric in metrics
    ) / total_examples
    train_accuracy = sum(
        num_examples * float(metric["train_accuracy"])
        for num_examples, metric in metrics
    ) / total_examples

    return {"train_loss": train_loss, "train_accuracy": train_accuracy}


def get_evaluate_fn(hidden_dim: int, dropout: float, run_logger: RunLogger):
    device = get_device()
    test_loader = get_test_loader(batch_size=256)
    criterion = nn.CrossEntropyLoss()

    def evaluate_global(server_round, parameters, config):
        model = ActivityMLP(hidden_dim=hidden_dim, dropout=dropout).to(device)
        set_model_parameters(model, parameters)
        loss, accuracy, confusion = evaluate(model, test_loader, criterion, device)
        macro_f1 = macro_f1_from_confusion(confusion)
        run_logger.log_metric(
            {
                "phase": "evaluate",
                "round": server_round,
                "loss": float(loss),
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
            }
        )
        return loss, {"accuracy": accuracy, "macro_f1": macro_f1}

    return evaluate_global


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        model_path: Path,
        run_logger: RunLogger,
        hidden_dim: int,
        dropout: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.run_logger = run_logger
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )
        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            save_model_parameters(
                ndarrays,
                self.model_path,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
            )
            print(f"Saved global model after round {server_round}: {self.model_path}")
            self.run_logger.log_metric(
                {
                    "phase": "fit",
                    "round": server_round,
                    "num_results": len(results),
                    "num_failures": len(failures),
                    "train_loss": aggregated_metrics.get("train_loss"),
                    "train_accuracy": aggregated_metrics.get("train_accuracy"),
                    "client_metrics": [
                        {
                            "num_examples": result.num_examples,
                            **dict(result.metrics),
                        }
                        for _, result in results
                    ],
                }
            )
        return aggregated_parameters, aggregated_metrics


def get_fit_config_fn(local_epochs: int):
    def fit_config(server_round: int):
        return {"local_epochs": local_epochs}

    return fit_config


def main():
    parser = argparse.ArgumentParser(description="Start the Flower HAR server.")
    parser.add_argument(
        "--server-address",
        type=str,
        default=env_str("SERVER_ADDRESS", "0.0.0.0:8080"),
        help="Address the Flower server should bind to.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=env_int("ROUNDS", 5),
        help="Federated rounds.",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=env_int("NUM_CLIENTS", 4),
        help="Expected clients.",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=env_int("LOCAL_EPOCHS", 1),
        help="Local epochs per client and round.",
    )
    parser.add_argument(
        "--client-batch-size",
        type=int,
        default=env_int("BATCH_SIZE", 64),
        help="Client batch size to record in the run config.",
    )
    parser.add_argument(
        "--client-lr",
        type=float,
        default=env_float("LR", 1e-3),
        help="Client learning rate to record in the run config.",
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
        "--run-id",
        type=str,
        default=env_str("RUN_ID", default_run_id()),
        help="Run id used for output files.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(env_str("RUNS_DIR", str(RUNS_DIR))),
        help="Directory for run outputs.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Where to save the aggregated global model.",
    )
    args = parser.parse_args()

    run_dir = args.runs_dir / args.run_id
    model_path = args.model_path or run_dir / "federated_global.pt"
    run_logger = RunLogger(run_dir)
    run_logger.write_config(
        {
            "run_id": args.run_id,
            "run_dir": run_dir,
            "server_address": args.server_address,
            "rounds": args.rounds,
            "num_clients": args.num_clients,
            "local_epochs": args.local_epochs,
            "batch_size": args.client_batch_size,
            "learning_rate": args.client_lr,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "model_path": model_path,
        }
    )
    print(f"Run directory: {run_dir}")

    torch.manual_seed(42)
    initial_model = ActivityMLP(hidden_dim=args.hidden_dim, dropout=args.dropout)
    initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

    strategy = SaveModelStrategy(
        model_path=model_path,
        run_logger=run_logger,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
        min_evaluate_clients=0,
        initial_parameters=initial_parameters,
        on_fit_config_fn=get_fit_config_fn(args.local_epochs),
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(args.hidden_dim, args.dropout, run_logger),
    )

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
