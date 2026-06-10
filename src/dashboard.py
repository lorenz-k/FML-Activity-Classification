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
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

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
            max_runs = st.number_input("Max Runs", min_value=1, max_value=100, value=24)
            submitted = st.form_submit_button("Start Batch")

        try:
            parsed = {
                "rounds": parse_int_values(rounds, "Rounds", 1, 50),
                "num_clients": parse_int_values(num_clients, "Clients", 1, 4),
                "local_epochs": parse_int_values(local_epochs, "Local Epochs", 1, 20),
                "learning_rate": parse_float_values(learning_rates, "Learning Rates", 0.00001, 1.0),
                "batch_size": parse_int_values(batch_sizes, "Batch Sizes", 1, 1024),
                "hidden_dim": parse_int_values(hidden_dims, "Hidden Dims", 1, 2048),
                "dropout": parse_float_values(dropouts, "Dropouts", 0.0, 0.8),
            }
            combinations = list(
                product(
                    parsed["rounds"],
                    parsed["num_clients"],
                    parsed["local_epochs"],
                    parsed["learning_rate"],
                    parsed["batch_size"],
                    parsed["hidden_dim"],
                    parsed["dropout"],
                )
            )
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
        for index, (r, clients, epochs, lr, bs, hd, do) in enumerate(combinations, start=1):
            run_id = (
                f"{clean_batch_id}_{index:02d}"
                f"_lr{slug_value(lr)}_hd{slug_value(hd)}_do{slug_value(do)}"
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


def main() -> None:
    st.title("FML HAR Dashboard")
    render_start_form()
    render_batch_form()
    render_batch_status()

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
