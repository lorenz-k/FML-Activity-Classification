# FML-Activity-Classification
Federated Machine Learning application for Human Activity Recognition (UCI HAR). Developed as part of the FML Programmierpraktikum at TU Berlin (SBE Group).
Topic B

## Code overview

The pipeline is file-based: each step writes artifacts into `outputs/` that the
next step reads. Roughly in the order things run:

- `src/data_preperation.py` – downloads UCI HAR and writes the client shards and
  the global test set.
- `src/local_model.py` – the MLP (`ActivityMLP`) used everywhere.
- `src/local_training.py` – trains the model on a single shard, without
  federation (a baseline to compare against).
- `src/flower_server.py` / `src/flower_client.py` – the federated setup: the
  server runs FedAvg rounds, each client trains on its own shard.
- `src/flower_utils.py`, `src/run_logging.py` – helpers for the parameter
  conversion Flower expects and for writing run configs and metrics.
- `src/experiment_runner.py` – starts a full server + clients run as background
  processes (and prepares the data partition first).
- `src/batch_runner.py` – runs a grid of experiment configs one after another.
- `src/dashboard.py` – Streamlit UI to start runs and inspect or compare results.

Every module is run as `python -m src.<module>` or through the dashboard.
`main.py` is only a placeholder, not an entry point.

## Dependency Management
This repository uses [uv](https://docs.astral.sh/uv/) for dependency management. Please install `uv` in your local shell then run: 
```bash
uv sync
```

This will create a virtual environment (.venv) with the specified python version and package versions, so we all develop with the same packages and don't run into any conflicts. To use the new environment/kernel in VS-Code, press `cmd-shift-P` (on mac, on windows `ctrl-shift-P`) and select a Python interpreter path. Since the jupyter notebooks are in the notebooks directory and not in the main project directory with the config files, you have to tell vscode to look there, so for the interpreter path enter `<your_local_path_to_FML-Activity-Classification>/.venv/bin/python`. Then click on Select kernel and choose the `fml-activity-classification` kernel.

## Data Preparation

Prepare the UCI HAR client shards and the global holdout test set:

```bash
uv run python src/data_preperation.py --n_clients 4 --seed 42
```

This creates:

```text
outputs/har/client_0.npz
outputs/har/client_1.npz
outputs/har/client_2.npz
outputs/har/client_3.npz
outputs/har/test_global.npz
outputs/har/meta.json
```

By default the client files are IID partitions of the official UCI HAR training
split. Note that this is IID with respect to the label classes, not with respect
to the subjects/people from the study in which the data was collected. The
official UCI HAR test split is kept separate as `test_global.npz`, and
`meta.json` records how the data was split (per-client sizes and class
distribution) so the dashboard can plot the layout of a run.

To simulate less friendly federated settings there are a few flags:

- `--partition-mode {iid,label_skew}` – `label_skew` uses a Dirichlet split so
  clients end up with skewed label distributions (non-IID).
- `--alpha` – Dirichlet concentration for `label_skew`. Smaller means stronger
  skew (e.g. `0.3`), larger is closer to IID.
- `--client-size-mode {balanced,imbalanced}` – `imbalanced` gives the first
  client the largest shard.
- `--data-dir` – write the partition somewhere other than `outputs/har`.

For example, a non-IID and imbalanced split:

```bash
uv run python src/data_preperation.py --n_clients 4 \
  --partition-mode label_skew --alpha 0.3 --client-size-mode imbalanced
```

For federated experiments started from the dashboard you usually don't run this
by hand: `experiment_runner` builds the requested partition on demand (under
`outputs/har/<layout>/`) and reuses it across runs.

## Local Model Training

Train the MLP on one client shard and evaluate on the global test set:

```bash
uv run python -m src.local_training --client-id 0 --epochs 20
```

The trained PyTorch state dict is saved to:

```text
outputs/models/local_client_0.pt
```

## Federated Training with Flower

Start one Flower server and four Flower clients in separate terminals:

```bash
uv run python -m src.flower_server --rounds 5 --num-clients 4 --run-id local_demo
```

```bash
uv run python -m src.flower_client --client-id 0
uv run python -m src.flower_client --client-id 1
uv run python -m src.flower_client --client-id 2
uv run python -m src.flower_client --client-id 3
```

The server coordinates FedAvg rounds. Each client trains on only its own
`outputs/har/client_<id>.npz` shard and sends model parameters back to the
server. The server also takes `--fraction-fit` and `--min-fit-clients` to train
on a subset of the clients per round, and records the partition settings in
`config.json` so the dashboard can show how a run was set up. The run config,
round metrics, and aggregated model are saved to:

```text
outputs/runs/local_demo/config.json
outputs/runs/local_demo/metrics.jsonl
outputs/runs/local_demo/federated_global.pt
```

`metrics.jsonl` is the file a later dashboard can poll or load for charts.
Each line is one JSON event, for example centralized evaluation metrics for one
federated round.

## Docker Compose Simulation

Build and run the deployment-like local setup with one server and four clients:

```bash
docker compose up --build
```

If your Docker installation uses the older standalone Compose binary, use:

```bash
docker-compose up --build
```

You can override experiment parameters with environment variables:

```bash
RUN_ID=docker_demo ROUNDS=10 LOCAL_EPOCHS=2 LR=0.0005 docker compose up --build
```

Inside Docker Compose, clients connect to the server through the service name:

```text
flower-server:8080
```

The `outputs` directory is mounted into every container, so run artifacts are
written back to the host machine:

```text
outputs/runs/<run_id>/config.json
outputs/runs/<run_id>/metrics.jsonl
outputs/runs/<run_id>/federated_global.pt
```

This Compose setup is the bridge toward Google Cloud: the same container image
and environment-variable configuration also run on Compute Engine VMs.

## Google Cloud

The cloud setup expects five Compute Engine VMs in the `fml-ac` project (zone
`europe-west3-a`): `fl-server` and `fl-client-0` … `fl-client-3`, plus the image
pushed to Artifact Registry. Both scripts run the server and clients over SSH,
one container per VM, and take the same environment-variable overrides as Docker
Compose.

`deploy.sh` pushes the local image first, then starts the VMs and the run:

```bash
sudo docker build -t fml-flower:local .
./deploy.sh
```

`run_cloud.sh` skips the image push (use it when the image is already in the
registry). It starts the VMs, runs the training, streams the server log, and
copies the finished run back into `outputs/runs/<run_id>` so the dashboard can
display it:

```bash
RUN_ID=gcp_demo ROUNDS=10 LR=0.0005 ./run_cloud.sh
```

Both need `gcloud` authenticated, and the VMs and registry have to exist
already.

## Dashboard

Start the Streamlit dashboard to inspect and compare saved runs:

```bash
uv run streamlit run src/dashboard.py
```

The dashboard reads:

```text
outputs/runs/<run_id>/config.json
outputs/runs/<run_id>/metrics.jsonl
```

It shows final metrics, per-round curves, client training metrics, and
comparisons across selected runs.

The sidebar can also start new local Flower runs, including the partition mode,
client sizes, participation, and alpha described under Data Preparation. It
launches `src.experiment_runner`, which starts one Flower server and the
selected number of clients as background processes. Runner status and logs are
written to:

```text
outputs/runs/<run_id>/status.json
outputs/runs/<run_id>/logs/server.log
outputs/runs/<run_id>/logs/client_<id>.log
```

The `Batch Runs` panel can start a small grid search. Enter comma-separated
values such as `0.001, 0.0005` for learning rates or `128, 256` for hidden
dimensions. The dashboard creates one run per parameter combination and starts
them sequentially through `src.batch_runner`. Batch status is stored in:

```text
outputs/batches/<batch_id>/configs.json
outputs/batches/<batch_id>/status.json
```

The sidebar also has a cloud panel. It shows the Compute Engine VM status, can
start and stop the VMs, and can launch a cloud run in the background (the same
steps as `run_cloud.sh`, generated per run under `outputs/cloud_runs/<run_id>`).
Finished cloud runs are synced back under `outputs/` and show up next to the
local ones.
