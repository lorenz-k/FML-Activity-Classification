import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent       ## __file__ 
RUNS_DIR = BASE_DIR / "outputs" / "runs"


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")

# we need this function bc pythons built-in json library (json.dump(dict)) only accepts strings and throws errors when inputing pyhton classes/objects
# here we recursively go through the dictionary and convert path objects to strings if we find some, and tuples to lists
def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return value


class RunLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)         # parents=true would also create parent dir, if they dont exist already
        self.metrics_path = self.run_dir / "metrics.jsonl"      # use jsonl (line) to easiyl append metrics iteratively after every round of FedAvg
        self.config_path = self.run_dir / "config.json"

    def write_config(self, config: dict[str, Any]) -> None:
        payload = json_safe(config)     # config with removed path objects, now safe for json.dumps (where s means string)
        self.config_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",   # indent=2 for pretty printing, 2 indents for every nesting level, alphabetic key order
            encoding="utf-8",
        )
        self.metrics_path.write_text("", encoding="utf-8")

    def log_metric(self, event: dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),    # use iso norm format to avoid confusion between 4/6/2026 and 6/4/2026
            **json_safe(event),     # ** avoids further nesting
        }
        with self.metrics_path.open("a", encoding="utf-8") as file:     # a means append for json-line appending of new rounds of metrics logged after one FedAvg round
            file.write(json.dumps(payload, sort_keys=True) + "\n")
