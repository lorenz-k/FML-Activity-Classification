import argparse
import json
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


def weighted_average(metrics):  # metrics is a list of the num_examples and metrics, aggregated cross-clients by flower
    # gets called by flower internally, gets called on (num_examples, metric_dict) subset of (model_params, num_exapmles, metric_dict) tuple that client's fit function returns
    # averages over train loss & acc that each client returns after training
    # weights each client's train loss/performance by the amount of training samples it had (important for settings where clients had different amount of train samples)
    total_examples = sum(num_examples for num_examples, _ in metrics)   # count/sum the num_expamples over the [metrics_client1, metrics_client2, metrics_client3, ...]
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


# define closure to pass evaluate_global fn with flower-defined params directly to server
# use closure to input more into evaluate_global with adding more input params than flower would expect
# also nicer than instantiate test_loader/criterion in main part then input
def get_evaluate_fn(hidden_dim: int, dropout: float, run_logger: RunLogger, data_dir=None):
    device = get_device()   # cpu most of the time, device≠client
    test_loader = get_test_loader(batch_size=256, data_dir=data_dir)       # loads pre-saved .npz with test data, returns torch Dataloader
    criterion = nn.CrossEntropyLoss()

    def evaluate_global(server_round, parameters, config):
        model = ActivityMLP(hidden_dim=hidden_dim, dropout=dropout).to(device)
        set_model_parameters(model, parameters)
        loss, accuracy, confusion = evaluate(model, test_loader, criterion, device) # just forward passes over test data + calculates accuracy and confusion mat
        # takes f1 for each class vs all others, then averages assigning same weight (irregardless of #samples in that class) to all classes when averaging
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


class SaveModelStrategy(fl.server.strategy.FedAvgM):
    # FedAvgM (Hsu et al., 2019): server-seitiges Momentum auf der Aggregation.
    # Mit server_momentum=0.0 reduziert sich das exakt auf FedAvg. Weil unsere
    # aggregate_fit nur super().aggregate_fit() aufruft und danach speichert/loggt,
    # kommt das Momentum "geschenkt" — die zurueckgegebenen Gewichte sind bereits
    # momentum-korrigiert.
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
        # result etc comes from clients and encompasses params, num_samples and metrics (used for weighted_avg)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(  # call/execute FedAvg from superclass 
            server_round,
            results,
            failures,
        )
        # aggregated_parameters is in flowers' own parameter format, not numpy, need to re-convert!
        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            save_model_parameters(      # saves averaged models as torch model
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


def get_fit_config_fn(local_epochs: int, proximal_mu: float = 0.0):
    # clients call fit_config at each round, and then get the same local_epochs number that we specified
    # becuase in the get_fit_config_fn the fit_config function is init with the same local_epochs (independent of server_round), this will stay consistent
    # we do this because flower framework works with such a function
    # could implement something more complicated here: do less local epochs (so on each client) the higher the round etc.
    # proximal_mu is the FedProx strength; broadcast so every client uses the same mu (0 = plain FedAvg)
    def fit_config(server_round: int):
        return {"local_epochs": local_epochs, "proximal_mu": proximal_mu}

    return fit_config


def main():
    parser = argparse.ArgumentParser(description="Start the Flower HAR server.")
    parser.add_argument(
        "--server-address",
        type=str,
        default=env_str("SERVER_ADDRESS", "0.0.0.0:8080"),  # server uses 0.0.0.0 not local host, so it listens to all incoming network interfaces/connections (from own and other devices), waits at 8080 port as standard
        help="Address & port the Flower server should bind to.",
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
        default=env_int("LOCAL_EPOCHS", 1),     # set low for initial testing
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
        "--fraction-fit",
        type=float,
        default=env_float("FRACTION_FIT", 1.0),
        help="Fraction of available clients sampled for training each round.",
    )
    parser.add_argument(
        "--min-fit-clients",
        type=int,
        default=env_int("MIN_FIT_CLIENTS", 0),
        help="Minimum clients trained per round (0 = all clients).",
    )
    parser.add_argument(
        "--partition-mode",
        type=str,
        default=env_str("PARTITION_MODE", "iid"),
        help="Data partition mode recorded in the run config (iid | label_skew).",
    )
    parser.add_argument(
        "--client-size-mode",
        type=str,
        default=env_str("CLIENT_SIZE_MODE", "balanced"),
        help="Client size mode recorded in the run config (balanced | imbalanced).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=env_float("ALPHA", 0.5),
        help="Dirichlet alpha recorded in the run config (label_skew only).",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=env_float("MU", 0.0),
        help="FedProx proximal term strength (0 = plain FedAvg).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=env_float("BETA", 0.0),
        help="FedAvgM server momentum (0 = plain FedAvg).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=env_str("FML_DATA_DIR", None),
        help="Directory holding the test set + meta.json (default: outputs/har).",
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
    model_path = args.model_path or run_dir / "federated_global.pt"     # final aggregated model
    run_logger = RunLogger(run_dir)     # imported from run_logging script

    # min_fit_clients == 0 means "use all clients" (keeps the old all-clients default)
    min_fit_clients = args.min_fit_clients or args.num_clients

    # label the run by which drift-mitigation is active (mu=client-side, beta=server-side)
    active = []
    if args.mu > 0.0:
        active.append("fedprox")
    if args.beta > 0.0:
        active.append("fedavgm")
    algorithm = "+".join(active) if active else "fedavg"

    # if a partition meta.json exists, pull its client sizes / class distribution into
    # the run config so the dashboard can plot the data layout for this run
    partition_meta = {}
    if args.data_dir:
        meta_path = Path(args.data_dir) / "meta.json"
        if meta_path.exists():
            partition_meta = json.loads(meta_path.read_text(encoding="utf-8"))

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
            # federated setup dimensions
            "partition_mode": args.partition_mode,
            "client_size_mode": args.client_size_mode,
            "alpha": args.alpha,
            "fraction_fit": args.fraction_fit,
            "min_fit_clients": min_fit_clients,
            "mu": args.mu,
            "beta": args.beta,
            "algorithm": algorithm,
            "data_dir": args.data_dir,
            # data layout (from meta.json if available)
            "client_sizes": partition_meta.get("client_sizes"),
            "class_distribution": partition_meta.get("class_distribution"),
        }
    )
    print(f"Run directory: {run_dir}")

    torch.manual_seed(42)
    initial_model = ActivityMLP(hidden_dim=args.hidden_dim, dropout=args.dropout)
    # get model parameters/weights (in pytorch statedict) -> list of params as numpy arrays
    # because flower expects numpy for weights
    initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

    strategy = SaveModelStrategy(
        model_path=model_path,
        run_logger=run_logger,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        fraction_fit=args.fraction_fit,     # fraction of available clients sampled per round (1.0 = all)
        fraction_evaluate=0.0,      # no client-specific eval, just on global model
        min_fit_clients=min_fit_clients,    # how many clients actually train per round
        min_available_clients=args.num_clients,     # wait until all clients connected, then sample
        min_evaluate_clients=0,
        initial_parameters=initial_parameters,
        server_momentum=args.beta,  # FedAvgM server momentum (0.0 = plain FedAvg)
        on_fit_config_fn=get_fit_config_fn(args.local_epochs, args.mu),  # broadcasts local_epochs + FedProx mu to clients each round
        fit_metrics_aggregation_fn=weighted_average,    # how to weigh clients performances to aggregate
        evaluate_fn=get_evaluate_fn(args.hidden_dim, args.dropout, run_logger, args.data_dir),     # evalute fn for the server
    )

    # built-in flower server func expects custom strategy class where we can add our own logging and other functionalities
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
