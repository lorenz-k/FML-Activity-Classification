import json
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "outputs" / "runs"
BATCHES_DIR = BASE_DIR / "outputs" / "batches"
CLOUD_RUNS_DIR = BASE_DIR / "outputs" / "cloud_runs"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CLOUD_PROJECT = "fml-ac"
CLOUD_ZONE = "europe-west3-a"
CLOUD_REGION = "europe-west3"
CLOUD_IMAGE = f"{CLOUD_REGION}-docker.pkg.dev/{CLOUD_PROJECT}/{CLOUD_PROJECT}/fml-flower:latest"
CLOUD_SERVER_IP = "10.156.0.2"
CLOUD_VMS = ["fl-server", "fl-client-0", "fl-client-1", "fl-client-2", "fl-client-3"]

from src.run_logging import default_run_id


st.set_page_config(
    page_title="FML HAR Dashboard",
    layout="wide",
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text_tail(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def slug_value(value: Any) -> str:
    text = str(value).replace(".", "p").replace("-", "m")
    return "".join(char if char.isalnum() else "_" for char in text)


def min_fit_from_fraction(fraction_fit: float, num_clients: int) -> int:
    # how many clients actually train per round given a participation fraction.
    return max(1, round(num_clients * fraction_fit))


def parse_int_values(raw: str, label: str, min_value: int, max_value: int) -> list[int]:
    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{label} must contain integers.") from exc
    if not values:
        raise ValueError(f"{label} needs at least one value.")
    invalid = [value for value in values if value < min_value or value > max_value]
    if invalid:
        raise ValueError(f"{label} values out of range: {invalid}")
    return values


def parse_float_values(raw: str, label: str, min_value: float, max_value: float) -> list[float]:
    try:
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{label} must contain numbers.") from exc
    if not values:
        raise ValueError(f"{label} needs at least one value.")
    invalid = [value for value in values if value < min_value or value > max_value]
    if invalid:
        raise ValueError(f"{label} values out of range: {invalid}")
    return values


def discover_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(
        [path for path in RUNS_DIR.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def discover_batches() -> list[Path]:
    if not BATCHES_DIR.exists():
        return []
    return sorted(
        [path for path in BATCHES_DIR.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def load_run(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    config = read_json(run_dir / "config.json")
    events = read_jsonl(run_dir / "metrics.jsonl")
    metrics = pd.DataFrame(events)
    if not metrics.empty and "timestamp" in metrics:
        metrics = metrics.sort_values("timestamp")
    return config, metrics


def load_status(run_dir: Path) -> dict[str, Any]:
    return read_json(run_dir / "status.json")


def start_local_run(config: dict[str, Any]) -> None:
    run_dir = RUNS_DIR / config["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = (run_dir / "runner_launcher.log").open("a", encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        "src.experiment_runner",
        "--run-id",
        config["run_id"],
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
        "--partition-mode",
        str(config.get("partition_mode", "iid")),
        "--client-size-mode",
        str(config.get("client_size_mode", "balanced")),
        "--alpha",
        str(config.get("alpha", 0.5)),
        "--fraction-fit",
        str(config.get("fraction_fit", 1.0)),
        "--min-fit-clients",
        str(config.get("min_fit_clients", 0)),
        "--seed",
        str(config.get("seed", 42)),
    ]

    subprocess.Popen(
        command,
        cwd=BASE_DIR,
        stdout=launcher_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    launcher_log.close()


def start_batch(batch_id: str, configs: list[dict[str, Any]]) -> None:
    batch_dir = BATCHES_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    configs_path = batch_dir / "configs.json"
    configs_path.write_text(json.dumps(configs, indent=2) + "\n", encoding="utf-8")
    launcher_log = (batch_dir / "launcher.log").open("a", encoding="utf-8")

    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "src.batch_runner",
            "--batch-id",
            batch_id,
            "--configs-path",
            str(configs_path),
        ],
        cwd=BASE_DIR,
        stdout=launcher_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    launcher_log.close()


def delete_run(run_dir: Path) -> None:
    if run_dir.exists() and run_dir.is_dir():
        shutil.rmtree(run_dir)


# --- Cloud helpers ---

@st.cache_data(ttl=20)
def get_vm_statuses() -> dict[str, str]:
    try:
        result = subprocess.run(
            [
                "gcloud", "compute", "instances", "list",
                "--project", CLOUD_PROJECT,
                "--zones", CLOUD_ZONE,
                "--filter", "name:(" + " ".join(CLOUD_VMS) + ")",
                "--format", "csv(name,status)",
            ],
            capture_output=True, text=True, timeout=15,
        )
        statuses: dict[str, str] = {}
        for line in result.stdout.strip().splitlines()[1:]:
            if "," in line:
                name, status = line.split(",", 1)
                statuses[name.strip()] = status.strip()
        return statuses
    except Exception:
        return {}


def start_cloud_run_background(config: dict[str, Any]) -> None:
    run_id = config["run_id"]
    cloud_run_dir = CLOUD_RUNS_DIR / run_id
    cloud_run_dir.mkdir(parents=True, exist_ok=True)

    script = f"""#!/bin/bash
set -e
PROJECT="{CLOUD_PROJECT}"
ZONE="{CLOUD_ZONE}"
REGION="{CLOUD_REGION}"
IMAGE="{CLOUD_IMAGE}"
SERVER_IP="{CLOUD_SERVER_IP}"
RUN_ID="{run_id}"
ROUNDS="{config['rounds']}"
NUM_CLIENTS="{config['num_clients']}"
LOCAL_EPOCHS="{config['local_epochs']}"
BATCH_SIZE="{config['batch_size']}"
LR="{config['learning_rate']}"
HIDDEN_DIM="{config['hidden_dim']}"
DROPOUT="{config['dropout']}"
PARTITION_MODE="{config.get('partition_mode', 'iid')}"
CLIENT_SIZE_MODE="{config.get('client_size_mode', 'balanced')}"
ALPHA="{config.get('alpha', 0.5)}"
FRACTION_FIT="{config.get('fraction_fit', 1.0)}"
MIN_FIT_CLIENTS="{config.get('min_fit_clients', 0)}"
SEED="{config.get('seed', 42)}"
LOCAL_RUNS="{RUNS_DIR}"

echo "[$(date -u +%H:%M:%S)] Starting server on fl-server..."
gcloud compute ssh fl-server --zone=$ZONE --project=$PROJECT --command="
  sudo docker stop fl-server 2>/dev/null || true
  sudo docker rm fl-server 2>/dev/null || true
  gcloud auth print-access-token | sudo docker login -u oauth2accesstoken --password-stdin $REGION-docker.pkg.dev
  sudo docker pull $IMAGE
  sudo docker run -d --name fl-server \\
    -p 8080:8080 \\
    -e RUN_ID=$RUN_ID -e ROUNDS=$ROUNDS -e NUM_CLIENTS=$NUM_CLIENTS \\
    -e LOCAL_EPOCHS=$LOCAL_EPOCHS -e BATCH_SIZE=$BATCH_SIZE \\
    -e LR=$LR -e HIDDEN_DIM=$HIDDEN_DIM -e DROPOUT=$DROPOUT \\
    -e PARTITION_MODE=$PARTITION_MODE -e CLIENT_SIZE_MODE=$CLIENT_SIZE_MODE \\
    -e ALPHA=$ALPHA -e FRACTION_FIT=$FRACTION_FIT -e MIN_FIT_CLIENTS=$MIN_FIT_CLIENTS \\
    $IMAGE sh -c \\"python -m src.data_preperation --n_clients $NUM_CLIENTS --seed $SEED \\
      --partition-mode $PARTITION_MODE --client-size-mode $CLIENT_SIZE_MODE --alpha $ALPHA && \\
      python -m src.flower_server --server-address 0.0.0.0:8080\\"
  echo Server started.
"

echo "[$(date -u +%H:%M:%S)] Waiting 10s for server to be ready..."
sleep 10

echo "[$(date -u +%H:%M:%S)] Starting clients..."
for i in 0 1 2 3; do
  gcloud compute ssh fl-client-$i --zone=$ZONE --project=$PROJECT --command="
    sudo docker stop fl-client 2>/dev/null || true
    sudo docker rm fl-client 2>/dev/null || true
    gcloud auth print-access-token | sudo docker login -u oauth2accesstoken --password-stdin $REGION-docker.pkg.dev
    sudo docker pull $IMAGE
    sudo docker run -d --name fl-client \\
      -e CLIENT_ID=$i -e SERVER_ADDRESS=$SERVER_IP:8080 \\
      -e LOCAL_EPOCHS=$LOCAL_EPOCHS -e BATCH_SIZE=$BATCH_SIZE \\
      -e LR=$LR -e HIDDEN_DIM=$HIDDEN_DIM -e DROPOUT=$DROPOUT \\
      $IMAGE sh -c \\"python -m src.data_preperation --n_clients $NUM_CLIENTS --seed $SEED \\
        --partition-mode $PARTITION_MODE --client-size-mode $CLIENT_SIZE_MODE --alpha $ALPHA && \\
        python -m src.flower_client\\"
    echo Client $i started.
  " &
done
wait

echo "[$(date -u +%H:%M:%S)] All clients started. Waiting for training to complete..."
while true; do
  STATUS=$(gcloud compute ssh fl-server --zone=$ZONE --project=$PROJECT \\
    --command="sudo docker inspect -f '{{{{.State.Status}}}}' fl-server 2>/dev/null || echo not_found" 2>/dev/null || echo error)
  echo "[$(date -u +%H:%M:%S)] Server container: $STATUS"
  [ "$STATUS" = "exited" ] && break
  sleep 15
done

echo "[$(date -u +%H:%M:%S)] Training complete. Fetching outputs..."
gcloud compute ssh fl-server --zone=$ZONE --project=$PROJECT --command="
  sudo docker cp fl-server:/app/outputs/runs/$RUN_ID /tmp/$RUN_ID 2>/dev/null || true
  sudo chmod -R a+r /tmp/$RUN_ID 2>/dev/null || true
"
mkdir -p "$LOCAL_RUNS"
gcloud compute scp --recurse fl-server:/tmp/$RUN_ID "$LOCAL_RUNS/" \\
  --zone=$ZONE --project=$PROJECT && \\
  echo "[$(date -u +%H:%M:%S)] Outputs saved to $LOCAL_RUNS/$RUN_ID" || \\
  echo "[$(date -u +%H:%M:%S)] WARNING: Failed to copy outputs — use Fetch button in dashboard."

echo "[$(date -u +%H:%M:%S)] Cloud run complete."
"""

    script_path = cloud_run_dir / "run.sh"
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)

    log_file = (cloud_run_dir / "run.log").open("a", encoding="utf-8")
    subprocess.Popen(
        ["bash", str(script_path)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()


def discover_cloud_runs() -> list[Path]:
    if not CLOUD_RUNS_DIR.exists():
        return []
    return sorted(
        [p for p in CLOUD_RUNS_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def latest_round_events(metrics: pd.DataFrame, phase: str) -> pd.DataFrame:
    if metrics.empty or "phase" not in metrics or "round" not in metrics:
        return pd.DataFrame()

    events = metrics[metrics["phase"] == phase].copy()
    if events.empty:
        return events

    if "timestamp" in events:
        events = events.sort_values("timestamp")

    return events.drop_duplicates(subset=["round"], keep="last").sort_values("round")


def final_eval_row(metrics: pd.DataFrame) -> dict[str, Any]:
    evaluations = latest_round_events(metrics, "evaluate")
    if evaluations.empty:
        return {}
    return evaluations.iloc[-1].to_dict()


def metric_card(label: str, value: Any, fmt: str = "{:.4f}") -> None:
    if value is None or pd.isna(value):
        st.metric(label, "n/a")
        return
    if isinstance(value, int):
        st.metric(label, str(value))
        return
    st.metric(label, fmt.format(float(value)))


def render_federated_setup(config: dict[str, Any]) -> None:
    # small metrics row + client data-layout plots for this run
    partition_mode = config.get("partition_mode")
    if partition_mode is None:
        return  # older run without the new fields

    client_size_mode = config.get("client_size_mode", "n/a")
    fraction_fit = config.get("fraction_fit")
    min_fit = config.get("min_fit_clients")
    num_clients = config.get("num_clients")

    st.markdown("**Federated Setup**")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Partition", "IID" if partition_mode == "iid" else "Non-IID")
    with cols[1]:
        st.metric("Client Sizes", str(client_size_mode).capitalize())
    with cols[2]:
        if fraction_fit is not None and num_clients:
            st.metric("Participation", f"{int(fraction_fit * 100)}% ({min_fit}/{num_clients})")
        else:
            st.metric("Participation", "n/a")
    with cols[3]:
        st.metric("Alpha", config.get("alpha") if partition_mode != "iid" else "—")

    client_sizes = config.get("client_sizes")
    class_distribution = config.get("class_distribution")

    plot_cols = st.columns(2)
    with plot_cols[0]:
        if client_sizes:
            st.caption("Samples per Client")
            sizes_df = pd.DataFrame(
                {"samples": client_sizes},
                index=[f"client_{i}" for i in range(len(client_sizes))],
            )
            st.bar_chart(sizes_df, height=260)
    with plot_cols[1]:
        if class_distribution:
            st.caption("Class Distribution per Client")
            # rows = clients, columns = activity labels -> stacked bars
            dist_df = pd.DataFrame(class_distribution).T.fillna(0)
            st.bar_chart(dist_df, height=260)


def render_run_detail(run_dir: Path, config: dict[str, Any], metrics: pd.DataFrame) -> None:
    st.subheader(run_dir.name)

    status = load_status(run_dir)
    final_eval = final_eval_row(metrics)
    top = st.columns(6)
    with top[0]:
        st.metric("Status", status.get("state", "n/a"))
    with top[1]:
        metric_card("Rounds", config.get("rounds"), "{:.0f}")
    with top[2]:
        metric_card("Clients", config.get("num_clients"), "{:.0f}")
    with top[3]:
        metric_card("Accuracy", final_eval.get("accuracy"))
    with top[4]:
        metric_card("Macro F1", final_eval.get("macro_f1"))
    with top[5]:
        metric_card("Loss", final_eval.get("loss"))

    config_cols = st.columns(4)
    with config_cols[0]:
        st.caption("Local Epochs")
        st.write(config.get("local_epochs", "n/a"))
    with config_cols[1]:
        st.caption("Learning Rate")
        st.write(config.get("learning_rate", "n/a"))
    with config_cols[2]:
        st.caption("Batch Size")
        st.write(config.get("batch_size", "n/a"))
    with config_cols[3]:
        st.caption("Dropout")
        st.write(config.get("dropout", "n/a"))

    render_federated_setup(config)

    if metrics.empty:
        st.warning("No metrics.jsonl events found for this run.")
        return

    evaluate = latest_round_events(metrics, "evaluate")
    fit = latest_round_events(metrics, "fit")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        if not evaluate.empty:
            st.line_chart(
                evaluate.set_index("round")[["accuracy", "macro_f1"]],
                height=280,
            )
    with chart_cols[1]:
        if not evaluate.empty:
            st.line_chart(evaluate.set_index("round")[["loss"]], height=280)

    if not fit.empty:
        st.markdown("**Training Metrics**")
        st.line_chart(
            fit.set_index("round")[["train_accuracy", "train_loss"]],
            height=260,
        )

    with st.expander("Raw Events"):
        st.dataframe(metrics, use_container_width=True, hide_index=True)

    logs_dir = run_dir / "logs"
    if logs_dir.exists() or (run_dir / "runner_launcher.log").exists():
        with st.expander("Logs"):
            server_log = read_text_tail(logs_dir / "server.log")
            if server_log:
                st.markdown("**Server**")
                st.code(server_log)
            for client_log in sorted(logs_dir.glob("client_*.log")):
                text = read_text_tail(client_log)
                if text:
                    st.markdown(f"**{client_log.stem}**")
                    st.code(text)
            launcher_log = read_text_tail(run_dir / "runner_launcher.log")
            if launcher_log:
                st.markdown("**Launcher**")
                st.code(launcher_log)


def render_comparison(selected_runs: list[Path]) -> None:
    rows = []
    for run_dir in selected_runs:
        config, metrics = load_run(run_dir)
        final_eval = final_eval_row(metrics)
        status = load_status(run_dir)
        rows.append(
            {
                "run_id": run_dir.name,
                "status": status.get("state"),
                "partition_mode": config.get("partition_mode"),
                "client_size_mode": config.get("client_size_mode"),
                "alpha": config.get("alpha"),
                "fraction_fit": config.get("fraction_fit"),
                "min_fit_clients": config.get("min_fit_clients"),
                "rounds": config.get("rounds"),
                "clients": config.get("num_clients"),
                "local_epochs": config.get("local_epochs"),
                "lr": config.get("learning_rate"),
                "batch_size": config.get("batch_size"),
                "dropout": config.get("dropout"),
                "accuracy": final_eval.get("accuracy"),
                "macro_f1": final_eval.get("macro_f1"),
                "loss": final_eval.get("loss"),
            }
        )

    comparison = pd.DataFrame(rows)
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    if len(selected_runs) < 2:
        return

    combined = []
    for run_dir in selected_runs:
        _, metrics = load_run(run_dir)
        if metrics.empty or "phase" not in metrics:
            continue
        evaluate = latest_round_events(metrics, "evaluate")
        if evaluate.empty:
            continue
        evaluate["run_id"] = run_dir.name
        combined.append(evaluate)

    if not combined:
        return

    combined_df = pd.concat(combined, ignore_index=True)
    accuracy_chart = combined_df.pivot_table(
        index="round",
        columns="run_id",
        values="accuracy",
        aggfunc="last",
    )
    f1_chart = combined_df.pivot_table(
        index="round",
        columns="run_id",
        values="macro_f1",
        aggfunc="last",
    )

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Accuracy Comparison**")
        st.line_chart(accuracy_chart, height=280)
    with cols[1]:
        st.markdown("**Macro-F1 Comparison**")
        st.line_chart(f1_chart, height=280)


def render_start_form() -> None:
    if "next_run_id" not in st.session_state:
        st.session_state["next_run_id"] = default_run_id()

    with st.sidebar.form("start_run"):
        st.markdown("**New Run**")
        run_id = st.text_input("Run ID", value=st.session_state["next_run_id"])
        rounds = st.number_input("Rounds", min_value=1, max_value=50, value=5, step=1)
        num_clients = st.number_input("Clients", min_value=1, max_value=4, value=4, step=1)
        local_epochs = st.number_input(
            "Local Epochs",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
        )
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.00001,
            max_value=1.0,
            value=0.001,
            step=0.0001,
            format="%.5f",
        )
        batch_size = st.selectbox("Batch Size", options=[16, 32, 64, 128, 256], index=2)
        hidden_dim = st.selectbox("Hidden Dim", options=[64, 128, 256, 512], index=1)
        dropout = st.slider("Dropout", min_value=0.0, max_value=0.8, value=0.2, step=0.05)

        st.markdown("**Federated Setup**")
        partition_label = st.selectbox(
            "Partition Mode", options=["IID", "Non-IID (label skew)"], index=0
        )
        partition_mode = "iid" if partition_label.startswith("IID") else "label_skew"
        client_size_label = st.selectbox(
            "Client Sizes", options=["Balanced", "Imbalanced"], index=0
        )
        client_size_mode = "balanced" if client_size_label == "Balanced" else "imbalanced"
        alpha = st.selectbox(
            "Non-IID Alpha", options=[0.3, 0.5, 1.0], index=1,
            help="Only used for label skew. Smaller = stronger non-IID.",
        )
        participation = st.selectbox("Participation", options=["100%", "50%"], index=0)
        fraction_fit = 1.0 if participation == "100%" else 0.5

        overwrite = st.checkbox("Overwrite existing run", value=False)
        submitted = st.form_submit_button("Start Run")

    if not submitted:
        return

    if not run_id.strip():
        st.sidebar.error("Run ID is required.")
        return

    run_dir = RUNS_DIR / run_id.strip()
    if run_dir.exists() and not overwrite:
        st.sidebar.error("Run ID already exists.")
        return

    start_local_run(
        {
            "run_id": run_id.strip(),
            "rounds": int(rounds),
            "num_clients": int(num_clients),
            "local_epochs": int(local_epochs),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "partition_mode": partition_mode,
            "client_size_mode": client_size_mode,
            "alpha": float(alpha),
            "fraction_fit": float(fraction_fit),
            "min_fit_clients": min_fit_from_fraction(fraction_fit, int(num_clients)),
        }
    )
    st.session_state["next_run_id"] = default_run_id()
    st.sidebar.success(f"Started {run_id.strip()}.")
    st.rerun()


def render_batch_form() -> None:
    if "next_batch_id" not in st.session_state:
        st.session_state["next_batch_id"] = default_run_id().replace("run_", "batch_")

    with st.sidebar.expander("Batch Runs"):
        with st.form("batch_runs"):
            batch_id = st.text_input("Batch ID", value=st.session_state["next_batch_id"])
            rounds = st.text_input("Rounds", value="5")
            num_clients = st.text_input("Clients", value="4")
            local_epochs = st.text_input("Local Epochs", value="1")
            learning_rates = st.text_input("Learning Rates", value="0.001, 0.0005")
            batch_sizes = st.text_input("Batch Sizes", value="64")
            hidden_dims = st.text_input("Hidden Dims", value="128, 256")
            dropouts = st.text_input("Dropouts", value="0.1, 0.2")
            st.markdown("**Federated Setup**")
            partition_modes = st.multiselect(
                "Partition Modes", options=["iid", "label_skew"], default=["iid"],
            )
            client_size_modes = st.multiselect(
                "Client Size Modes", options=["balanced", "imbalanced"], default=["balanced"],
            )
            participation_rates = st.multiselect(
                "Participation Rates", options=[1.0, 0.5], default=[1.0],
            )
            alphas = st.text_input("Alphas (label skew)", value="0.3, 1.0")
            max_runs = st.number_input("Max Runs", min_value=1, max_value=200, value=24)
            submitted = st.form_submit_button("Start Batch")

        try:
            if not partition_modes:
                raise ValueError("Select at least one partition mode.")
            if not client_size_modes:
                raise ValueError("Select at least one client size mode.")
            if not participation_rates:
                raise ValueError("Select at least one participation rate.")
            parsed = {
                "rounds": parse_int_values(rounds, "Rounds", 1, 50),
                "num_clients": parse_int_values(num_clients, "Clients", 1, 4),
                "local_epochs": parse_int_values(local_epochs, "Local Epochs", 1, 20),
                "learning_rate": parse_float_values(learning_rates, "Learning Rates", 0.00001, 1.0),
                "batch_size": parse_int_values(batch_sizes, "Batch Sizes", 1, 1024),
                "hidden_dim": parse_int_values(hidden_dims, "Hidden Dims", 1, 2048),
                "dropout": parse_float_values(dropouts, "Dropouts", 0.0, 0.8),
                "alpha": parse_float_values(alphas, "Alphas", 0.01, 100.0),
            }
            combinations = []
            seen = set()
            for combo in product(
                parsed["rounds"],
                parsed["num_clients"],
                parsed["local_epochs"],
                parsed["learning_rate"],
                parsed["batch_size"],
                parsed["hidden_dim"],
                parsed["dropout"],
                partition_modes,
                client_size_modes,
                participation_rates,
                parsed["alpha"],
            ):
                pmode, alpha = combo[7], combo[10]
                # alpha is meaningless for IID — collapse to one alpha to avoid dupes
                key = combo[:10] + ((None,) if pmode == "iid" else (alpha,))
                if key in seen:
                    continue
                seen.add(key)
                combinations.append(combo)
            st.caption(f"{len(combinations)} run combinations")
        except ValueError as exc:
            st.error(str(exc))
            return

        if not submitted:
            return

        clean_batch_id = batch_id.strip()
        if not clean_batch_id:
            st.error("Batch ID is required.")
            return
        if len(combinations) > int(max_runs):
            st.error(f"Batch has {len(combinations)} runs, max is {int(max_runs)}.")
            return

        configs = []
        for index, combo in enumerate(combinations, start=1):
            r, clients, epochs, lr, bs, hd, do, pmode, csize, frac, alpha = combo
            part_tag = "iid" if pmode == "iid" else f"noniid_a{slug_value(alpha)}"
            run_id = (
                f"{clean_batch_id}_{index:02d}_{part_tag}_{csize}"
                f"_part{int(frac * 100)}_lr{slug_value(lr)}_hd{slug_value(hd)}"
            )
            configs.append(
                {
                    "run_id": run_id,
                    "rounds": int(r),
                    "num_clients": int(clients),
                    "local_epochs": int(epochs),
                    "learning_rate": float(lr),
                    "batch_size": int(bs),
                    "hidden_dim": int(hd),
                    "dropout": float(do),
                    "partition_mode": pmode,
                    "client_size_mode": csize,
                    "alpha": float(alpha),
                    "fraction_fit": float(frac),
                    "min_fit_clients": min_fit_from_fraction(frac, int(clients)),
                }
            )

        existing = [config["run_id"] for config in configs if (RUNS_DIR / config["run_id"]).exists()]
        if existing:
            st.error(f"Run IDs already exist: {', '.join(existing[:5])}")
            return

        start_batch(clean_batch_id, configs)
        st.session_state["next_batch_id"] = default_run_id().replace("run_", "batch_")
        st.success(f"Started batch {clean_batch_id} with {len(configs)} runs.")
        st.rerun()


def render_batch_status() -> None:
    batches = discover_batches()
    if not batches:
        return

    with st.sidebar.expander("Batch Status"):
        for batch_dir in batches[:5]:
            status = read_json(batch_dir / "status.json")
            state = status.get("state", "queued")
            completed = status.get("completed_runs", 0)
            failed = status.get("failed_runs", 0)
            total = status.get("total_runs", "?")
            current = status.get("current_run")
            st.caption(f"{batch_dir.name}: {state} ({completed}/{total}, failed {failed})")
            if current:
                st.caption(f"current: {current}")


def render_run_picker(runs: list[Path]) -> list[Path]:
    run_names = [path.name for path in runs]
    selected_names = st.sidebar.multiselect(
        "Runs",
        options=run_names,
        default=run_names[: min(3, len(run_names))],
    )
    selected_runs = [path for path in runs if path.name in selected_names]

    for run_name in selected_names:
        cols = st.sidebar.columns([0.78, 0.22])
        cols[0].caption(run_name)
        clicked = cols[1].button(
            "Del",
            key=f"delete_{run_name}",
            type="secondary",
            use_container_width=True,
        )
        if not clicked:
            continue

        if st.session_state.get("pending_delete_run") == run_name:
            selected_run = next(path for path in runs if path.name == run_name)
            status = load_status(selected_run)
            if status.get("state") in {"starting", "running"}:
                st.sidebar.error("Active runs are protected.")
                return selected_runs
            delete_run(selected_run)
            st.session_state.pop("pending_delete_run", None)
            st.sidebar.success(f"Deleted {run_name}.")
            st.rerun()

        st.session_state["pending_delete_run"] = run_name
        st.sidebar.warning(f"Click Del again to confirm {run_name}.")

    return selected_runs


def render_cloud_sidebar() -> None:
    with st.sidebar.expander("Cloud (GCP)"):
        hdr, btn = st.columns([3, 1])
        hdr.markdown("**VM Status**")
        if btn.button("↻", key="refresh_vms", help="Refresh"):
            get_vm_statuses.clear()
            st.rerun()

        statuses = get_vm_statuses()
        icons = {"RUNNING": "🟢", "TERMINATED": "🔴", "STAGING": "🟡", "STOPPING": "🟡"}
        if statuses:
            for vm in CLOUD_VMS:
                s = statuses.get(vm, "?")
                st.caption(f"{icons.get(s, '⚫')} {vm}: {s}")
        else:
            st.caption("Could not reach gcloud")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Start VMs", key="btn_start_vms", use_container_width=True):
                subprocess.Popen(
                    ["gcloud", "compute", "instances", "start"] + CLOUD_VMS
                    + ["--zone", CLOUD_ZONE, "--project", CLOUD_PROJECT],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                get_vm_statuses.clear()
                st.success("Starting VMs (~30s)...")
        with c2:
            if st.button("Stop VMs", key="btn_stop_vms", use_container_width=True):
                subprocess.Popen(
                    ["gcloud", "compute", "instances", "stop"] + CLOUD_VMS
                    + ["--zone", CLOUD_ZONE, "--project", CLOUD_PROJECT],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                get_vm_statuses.clear()
                st.success("Stopping VMs...")

        st.divider()

        if "next_cloud_run_id" not in st.session_state:
            st.session_state["next_cloud_run_id"] = "gcp_" + default_run_id()

        with st.form("cloud_run_form"):
            st.markdown("**New Cloud Run**")
            run_id = st.text_input("Run ID", value=st.session_state["next_cloud_run_id"])
            rounds = st.number_input("Rounds", min_value=1, max_value=50, value=5, step=1)
            num_clients = st.number_input("Clients", min_value=1, max_value=4, value=4, step=1)
            local_epochs = st.number_input("Local Epochs", min_value=1, max_value=20, value=1, step=1)
            learning_rate = st.number_input(
                "Learning Rate", min_value=0.00001, max_value=1.0,
                value=0.001, step=0.0001, format="%.5f",
            )
            batch_size = st.selectbox("Batch Size", options=[16, 32, 64, 128, 256], index=2)
            hidden_dim = st.selectbox("Hidden Dim", options=[64, 128, 256, 512], index=1)
            dropout = st.slider("Dropout", min_value=0.0, max_value=0.8, value=0.2, step=0.05)
            partition_label = st.selectbox(
                "Partition Mode", options=["IID", "Non-IID (label skew)"], index=0,
                key="cloud_partition",
            )
            client_size_label = st.selectbox(
                "Client Sizes", options=["Balanced", "Imbalanced"], index=0,
                key="cloud_client_size",
            )
            alpha = st.selectbox("Non-IID Alpha", options=[0.3, 0.5, 1.0], index=1, key="cloud_alpha")
            participation = st.selectbox("Participation", options=["100%", "50%"], index=0, key="cloud_part")
            submitted = st.form_submit_button("Start Cloud Run")

        if submitted:
            if not run_id.strip():
                st.sidebar.error("Run ID is required.")
            else:
                partition_mode = "iid" if partition_label.startswith("IID") else "label_skew"
                client_size_mode = "balanced" if client_size_label == "Balanced" else "imbalanced"
                fraction_fit = 1.0 if participation == "100%" else 0.5
                start_cloud_run_background({
                    "run_id": run_id.strip(),
                    "rounds": int(rounds),
                    "num_clients": int(num_clients),
                    "local_epochs": int(local_epochs),
                    "learning_rate": float(learning_rate),
                    "batch_size": int(batch_size),
                    "hidden_dim": int(hidden_dim),
                    "dropout": float(dropout),
                    "partition_mode": partition_mode,
                    "client_size_mode": client_size_mode,
                    "alpha": float(alpha),
                    "fraction_fit": float(fraction_fit),
                    "min_fit_clients": min_fit_from_fraction(fraction_fit, int(num_clients)),
                })
                st.session_state["next_cloud_run_id"] = "gcp_" + default_run_id()
                st.sidebar.success(f"Cloud run '{run_id.strip()}' started.")
                st.rerun()


def render_cloud_tab() -> None:
    cloud_runs = discover_cloud_runs()
    if not cloud_runs:
        st.info("No cloud runs yet. Start one from the Cloud (GCP) section in the sidebar.")
        return

    selected_name = st.selectbox("Cloud Run", options=[p.name for p in cloud_runs])
    if not selected_name:
        return

    run_dir = next(p for p in cloud_runs if p.name == selected_name)
    log_path = run_dir / "run.log"

    hdr, btn = st.columns([4, 1])
    hdr.markdown(f"**Log: {selected_name}**")
    if btn.button("↻ Refresh", key="refresh_cloud_log"):
        st.rerun()

    log_text = read_text_tail(log_path, max_lines=100)
    if log_text:
        st.code(log_text, language="bash")
    else:
        st.caption("No log output yet — the background script may still be starting.")

    st.divider()

    local_run = RUNS_DIR / selected_name
    if local_run.exists():
        st.success(f"Outputs available — visible in 'Local Runs' tab as '{selected_name}'.")
    else:
        st.caption("Outputs are fetched automatically when training finishes. Use the button below if needed.")
        if st.button("Fetch Outputs from Server", key="fetch_outputs"):
            with st.spinner("Copying outputs from fl-server..."):
                try:
                    subprocess.run(
                        [
                            "gcloud", "compute", "ssh", "fl-server",
                            "--zone", CLOUD_ZONE, "--project", CLOUD_PROJECT,
                            "--command",
                            f"sudo docker cp fl-server:/app/outputs/runs/{selected_name} /tmp/{selected_name} 2>/dev/null; "
                            f"sudo chmod -R a+r /tmp/{selected_name} 2>/dev/null || true",
                        ],
                        check=False, capture_output=True, timeout=60,
                    )
                    RUNS_DIR.mkdir(parents=True, exist_ok=True)
                    subprocess.run(
                        [
                            "gcloud", "compute", "scp", "--recurse",
                            f"fl-server:/tmp/{selected_name}",
                            str(RUNS_DIR) + "/",
                            "--zone", CLOUD_ZONE, "--project", CLOUD_PROJECT,
                        ],
                        check=True, capture_output=True, timeout=120,
                    )
                    st.success("Outputs fetched. Switch to 'Local Runs' tab to view them.")
                    st.rerun()
                except subprocess.CalledProcessError as exc:
                    stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else str(exc.stderr)
                    st.error(f"Failed to fetch outputs: {stderr}")
                except subprocess.TimeoutExpired:
                    st.error("Timeout while fetching outputs.")


def main() -> None:
    st.title("FML HAR Dashboard")
    render_start_form()
    render_batch_form()
    render_batch_status()
    render_cloud_sidebar()

    local_tab, cloud_tab = st.tabs(["Runs", "Cloud Logs"])

    with cloud_tab:
        render_cloud_tab()

    with local_tab:
        runs = discover_runs()
        if not runs:
            st.info("No runs found yet. Start a Flower run to create outputs/runs/<run_id>.")
            return

        selected_runs = render_run_picker(runs)

        if not selected_runs:
            st.warning("Select at least one run.")
            return

        st.markdown("**Run Comparison**")
        render_comparison(selected_runs)

        st.divider()
        detail_name = st.selectbox("Run Detail", options=[path.name for path in selected_runs])
        detail_run = next(path for path in selected_runs if path.name == detail_name)
        config, metrics = load_run(detail_run)
        render_run_detail(detail_run, config, metrics)


if __name__ == "__main__":
    main()
