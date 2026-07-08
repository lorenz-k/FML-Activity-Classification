import argparse
import json
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.run_logging import RUNS_DIR, default_run_id, json_safe


BASE_DIR = Path(__file__).resolve().parent.parent
HAR_DIR = BASE_DIR / "outputs" / "har"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug_value(value: Any) -> str:
    text = str(value).replace(".", "p").replace("-", "m")
    return "".join(char if char.isalnum() else "_" for char in text)


def data_dir_for(partition_mode: str, client_size_mode: str, alpha: float, seed: int) -> Path:
    # one folder per data layout so partitions don't overwrite each other, e.g.
    # outputs/har/label_skew_imbalanced_alpha0p3_seed42/
    parts = [partition_mode, client_size_mode]
    if partition_mode != "iid":
        parts.append(f"alpha{slug_value(alpha)}")
    parts.append(f"seed{seed}")
    return HAR_DIR / "_".join(parts)


def ensure_data_prepared(
    data_dir: Path, num_clients: int, partition_mode: str,
    client_size_mode: str, alpha: float, seed: int, force: bool,
) -> None:
    # build the partition only if it's missing (or forced); reused across runs
    marker = data_dir / f"client_{num_clients - 1}.npz"
    if marker.exists() and not force:
        print(f"Data partition already present: {data_dir}")
        return
    print(f"Preparing data partition: {data_dir}")
    subprocess.run(
        [
            sys.executable, "-m", "src.data_preperation",
            "--n_clients", str(num_clients),
            "--seed", str(seed),
            "--partition-mode", partition_mode,
            "--client-size-mode", client_size_mode,
            "--alpha", str(alpha),
            "--data-dir", str(data_dir),
        ],
        cwd=BASE_DIR,
        check=True,
    )


def write_status(run_dir: Path, status: dict[str, Any]) -> None:
    status_path = run_dir / "status.json"
    status_path.write_text(
        json.dumps(json_safe(status), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_port(host: str, port: int, timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.25)
    return False


def open_log(run_dir: Path, name: str):
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return (logs_dir / name).open("w", encoding="utf-8")


def launch_process(command: list[str], log_file):
    return subprocess.Popen(
        command,
        cwd=BASE_DIR,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local Flower experiment.")
    parser.add_argument("--run-id", type=str, default=default_run_id())
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    # federated setup dimensions
    parser.add_argument("--partition-mode", choices=["iid", "label_skew"], default="iid")
    parser.add_argument("--client-size-mode", choices=["balanced", "imbalanced"], default="balanced")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--fraction-fit", type=float, default=1.0)
    parser.add_argument("--min-fit-clients", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Override the partition directory (default: derived from setup).")
    parser.add_argument("--skip-data-prep", action="store_true",
                        help="Do not (re)build the data partition before running.")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = args.runs_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # resolve the partition directory and make sure the shards exist
    data_dir = args.data_dir or data_dir_for(
        args.partition_mode, args.client_size_mode, args.alpha, args.seed
    )
    data_dir = Path(data_dir)
    if not args.skip_data_prep:
        ensure_data_prepared(
            data_dir, args.num_clients, args.partition_mode,
            args.client_size_mode, args.alpha, args.seed, force=False,
        )

    min_fit_clients = args.min_fit_clients or args.num_clients

    port = args.port or find_free_port()
    server_address = f"127.0.0.1:{port}"

    status = {
        "run_id": args.run_id,
        "state": "starting",
        "started_at": utc_now(),
        "finished_at": None,
        "server_address": server_address,
        "exit_code": None,
        "client_exit_codes": {},
        "error": None,
    }
    write_status(run_dir, status)

    ### LAUNCH SERVER

    server_log = open_log(run_dir, "server.log")
    client_logs = []
    processes = []

    server_command = [
        sys.executable,
        "-m",
        "src.flower_server",
        "--server-address",
        server_address,
        "--rounds",
        str(args.rounds),
        "--num-clients",
        str(args.num_clients),
        "--local-epochs",
        str(args.local_epochs),
        "--client-batch-size",
        str(args.batch_size),
        "--client-lr",
        str(args.lr),
        "--hidden-dim",
        str(args.hidden_dim),
        "--dropout",
        str(args.dropout),
        "--fraction-fit",
        str(args.fraction_fit),
        "--min-fit-clients",
        str(min_fit_clients),
        "--partition-mode",
        args.partition_mode,
        "--client-size-mode",
        args.client_size_mode,
        "--alpha",
        str(args.alpha),
        "--data-dir",
        str(data_dir),
        "--run-id",
        args.run_id,
        "--runs-dir",
        str(args.runs_dir),
    ]

    server_process = launch_process(server_command, server_log)
    processes.append(server_process)

    ## LAUNCH CLIENTS
    try:
        if not wait_for_port("127.0.0.1", port, args.startup_timeout):
            raise RuntimeError(f"Flower server did not open port {port}.")

        status["state"] = "running"
        write_status(run_dir, status)

        for client_id in range(args.num_clients):
            log_file = open_log(run_dir, f"client_{client_id}.log")
            client_logs.append(log_file)
            client_command = [
                sys.executable,
                "-m",
                "src.flower_client",
                "--client-id",
                str(client_id),
                "--server-address",
                server_address,
                "--epochs",
                str(args.local_epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--hidden-dim",
                str(args.hidden_dim),
                "--dropout",
                str(args.dropout),
                "--data-dir",
                str(data_dir),
            ]
            processes.append(launch_process(client_command, log_file))

        exit_code = server_process.wait()
        status["exit_code"] = exit_code
        status["client_exit_codes"] = {
            str(client_id): process.wait(timeout=10)
            for client_id, process in enumerate(processes[1:])
        }
        status["state"] = "completed" if exit_code == 0 else "failed"
        return exit_code

    except Exception as exc:
        status["state"] = "failed"
        status["error"] = str(exc)
        return 1

    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for log_file in [server_log, *client_logs]:
            log_file.close()
        status["finished_at"] = utc_now()
        write_status(run_dir, status)


if __name__ == "__main__":
    raise SystemExit(main())
