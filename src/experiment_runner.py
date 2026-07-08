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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = args.runs_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

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
