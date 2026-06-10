import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.run_logging import json_safe


BASE_DIR = Path(__file__).resolve().parent.parent
BATCHES_DIR = BASE_DIR / "outputs" / "batches"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def runner_command(config: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.experiment_runner",
        "--run-id",
        str(config["run_id"]),
        "--rounds",
        str(config["rounds"]),
        "--num-clients",
        str(config["num_clients"]),
        "--local-epochs",
        str(config["local_epochs"]),
        "--batch-size",
        str(config["batch_size"]),
        "--lr",
        str(config["learning_rate"]),
        "--hidden-dim",
        str(config["hidden_dim"]),
        "--dropout",
        str(config["dropout"]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multiple Flower experiments sequentially.")
    parser.add_argument("--batch-id", type=str, required=True)
    parser.add_argument("--configs-path", type=Path, required=True)
    args = parser.parse_args()

    batch_dir = BATCHES_DIR / args.batch_id
    status_path = batch_dir / "status.json"
    configs = json.loads(args.configs_path.read_text(encoding="utf-8"))

    status = {
        "batch_id": args.batch_id,
        "state": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "total_runs": len(configs),
        "completed_runs": 0,
        "failed_runs": 0,
        "current_run": None,
        "runs": [],
    }
    write_json(status_path, status)

    for config in configs:
        status["current_run"] = config["run_id"]
        status["runs"].append({"run_id": config["run_id"], "state": "running"})
        write_json(status_path, status)

        completed = subprocess.run(runner_command(config), cwd=BASE_DIR, check=False)
        run_state = "completed" if completed.returncode == 0 else "failed"
        status["runs"][-1]["state"] = run_state
        status["runs"][-1]["exit_code"] = completed.returncode
        if completed.returncode == 0:
            status["completed_runs"] += 1
        else:
            status["failed_runs"] += 1
        write_json(status_path, status)

    status["state"] = "completed" if status["failed_runs"] == 0 else "completed_with_failures"
    status["current_run"] = None
    status["finished_at"] = utc_now()
    write_json(status_path, status)
    return 0 if status["failed_runs"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
