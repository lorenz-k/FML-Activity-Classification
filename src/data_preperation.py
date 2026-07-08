"""
Federated Learning UCI HAR Data Preparation
==============================================

Outputs (./outputs/har/):
  client_0.npz … client_3.npz
  test_global.npz

Pipeline:
  1. load the official UCI HAR train and test splits
  2. partition only the official train split into N equal-ish client shards
  3. keep the official test split as a fixed global holdout set

Usage:
  python src/data_preperation.py                       # 4 clients
  python src/data_preperation.py --n_clients 4 --seed 42
"""

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

# paths
BASE_DIR    = Path(__file__).parent.parent
CACHE_DIR   = BASE_DIR / "cache"
DEFAULT_DIR = BASE_DIR / "outputs" / "har"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DIR.mkdir(parents=True, exist_ok=True)

# dataset constants
HAR_URL       = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/00240/UCI%20HAR%20Dataset.zip"
)
HAR_ZIP       = CACHE_DIR / "UCI_HAR.zip"
HAR_EXTRACT   = CACHE_DIR / "UCI HAR Dataset"
N_FEATURES    = 561
N_CLASSES     = 6

LABEL_MAP = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
}


# Download dataset

def download_har():
    # download if not cached already
    if HAR_ZIP.exists():
        print(f"Ordner existiert bei {Path(HAR_ZIP)}")
        return

    print(f"download Dataset …")

    req = urllib.request.Request(HAR_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = resp.read()
        HAR_ZIP.write_bytes(data)
        print()
        print(f"saved {HAR_ZIP.name}")
    except Exception as exc:
        raise RuntimeError(f"Download failed: {exc}") from exc


def extract_har():
    # unzip the har dataset
    if HAR_EXTRACT.exists():
        print(f"already extracted: {HAR_EXTRACT.name}/")
        return
    print("extract:  unzipping …")
    with zipfile.ZipFile(HAR_ZIP) as zf:
        zf.extractall(CACHE_DIR)

    ## remove annoying mac dir
    mac_dir = CACHE_DIR / "__MACOSX"
    if mac_dir.exists():
        shutil.rmtree(mac_dir)
        print("deleted __MACOSX/")

    print(f"done.  file ist here: {HAR_EXTRACT.name}/")


# Load & merge splits

def load_split(split: str):
    #load X and y from train or test directory
    d = HAR_EXTRACT / split
    X = np.loadtxt(d / f"X_{split}.txt")                      # (N, 561) float64
    y = np.loadtxt(d / f"y_{split}.txt", dtype=np.int64) - 1  # 1-based to 0-based indexing
    print(f"load  {split:5s}  X={X.shape}  y={y.shape}")
    return X, y


def validate_split(X: np.ndarray, y: np.ndarray, split: str):
    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise ValueError(f"{split} has invalid feature shape: {X.shape}")
    if y.ndim != 1 or len(X) != len(y):
        raise ValueError(f"{split} has incompatible X/y shapes: {X.shape}, {y.shape}")
    if y.min() < 0 or y.max() >= N_CLASSES:
        raise ValueError(f"{split} labels must be in range 0..{N_CLASSES - 1}")


def load_har_real():
    # Download, extract, and return official train/test splits separately.
    download_har()
    extract_har()

    X_tr, y_tr = load_split("train")
    X_te, y_te = load_split("test")
    validate_split(X_tr, y_tr, "train")
    validate_split(X_te, y_te, "test")

    print(f"keep test holdout: X={X_te.shape}  y={y_te.shape}")
    return X_tr, y_tr, X_te, y_te



# Client size weights

def client_fractions(client_size_mode: str, n_clients: int):
    # how many samples each client gets, as fractions of the train split.
    if client_size_mode == "balanced":
        return [1.0 / n_clients] * n_clients
    if client_size_mode == "imbalanced":
        if n_clients == 4:
            return [0.50, 0.25, 0.15, 0.10]
        # generic geometric decay for other client counts
        weights = np.array([0.5 ** i for i in range(n_clients)], dtype=float)
        return (weights / weights.sum()).tolist()
    raise ValueError(f"unknown client_size_mode: {client_size_mode}")


def target_sizes(fractions, n_total: int):
    # turn fractions into integer sample counts that sum exactly to n_total
    fr = np.array(fractions, dtype=float)
    fr = fr / fr.sum()
    sizes = np.floor(fr * n_total).astype(int)
    remainder = n_total - int(sizes.sum())
    for i in range(remainder):
        sizes[i % len(sizes)] += 1
    return sizes.tolist()


# Partitioning

def partition_iid(X: np.ndarray, y: np.ndarray, sizes, seed: int):
    # random shuffle, then hand out `sizes[i]` samples to each client.
    # every client ends up with a roughly global class distribution.
    idx = np.random.default_rng(seed).permutation(len(y))
    splits = []
    start = 0
    for size in sizes:
        sel = idx[start:start + size]
        splits.append((X[sel], y[sel]))
        start += size
    return splits


def partition_label_skew(X: np.ndarray, y: np.ndarray, sizes, alpha: float, seed: int):
    # Dirichlet label-skew partitioning (non-IID).
    # For each client we draw a class-proportion vector from Dirichlet(alpha):
    #   alpha large (e.g. 10) -> proportions near uniform -> almost IID
    #   alpha small (e.g. 0.3) -> proportions very peaked  -> strong non-IID
    # We then fill each client up to its target size by sampling (without
    # replacement) from per-class pools according to those proportions.
    rng = np.random.default_rng(seed)
    classes = list(range(N_CLASSES))

    pools = {k: list(rng.permutation(np.where(y == k)[0])) for k in classes}
    props = rng.dirichlet(alpha * np.ones(N_CLASSES), size=len(sizes))

    client_indices = [[] for _ in sizes]
    for c, size in enumerate(sizes):
        need = int(size)
        while need > 0:
            available = [k for k in classes if pools[k]]
            if not available:
                break
            p = np.array([props[c][k] for k in available], dtype=float)
            p = np.ones(len(available)) if p.sum() == 0 else p / p.sum()
            k = available[rng.choice(len(available), p=p)]
            client_indices[c].append(pools[k].pop())
            need -= 1

    # assign any leftover samples (pool exhaustion) round-robin so nothing is lost
    leftover = [idx for k in classes for idx in pools[k]]
    for i, idx in enumerate(leftover):
        client_indices[i % len(sizes)].append(idx)

    return [(X[np.array(ci)], y[np.array(ci)]) for ci in client_indices]


def make_partition(X, y, n_clients, partition_mode, client_size_mode, alpha, seed):
    fractions = client_fractions(client_size_mode, n_clients)
    sizes = target_sizes(fractions, len(y))
    if partition_mode == "iid":
        splits = partition_iid(X, y, sizes, seed)
    elif partition_mode == "label_skew":
        splits = partition_label_skew(X, y, sizes, alpha, seed)
    else:
        raise ValueError(f"unknown partition_mode: {partition_mode}")
    print(f"partition_mode={partition_mode}  client_size_mode={client_size_mode}  "
          f"alpha={alpha}")
    print(f"split data into {n_clients} clients, sizes = {[len(s[1]) for s in splits]}")
    return splits


def class_distribution(splits):
    dist = {}
    for i, (_, y) in enumerate(splits):
        counts = np.bincount(y, minlength=N_CLASSES)
        dist[f"client_{i}"] = {LABEL_MAP[k]: int(counts[k]) for k in range(N_CLASSES)}
    return dist


# save

def save_clients(splits, output_dir: Path):
    # write client_i.npz files
    for i, (X, y) in enumerate(splits):
        path = output_dir / f"client_{i}.npz"
        np.savez_compressed(path, X=X, y=y)
        print(f"saved {path.relative_to(BASE_DIR)}  X={X.shape}  y={y.shape}")


def save_test_set(X_test: np.ndarray, y_test: np.ndarray, output_dir: Path):
    path = output_dir / "test_global.npz"
    np.savez_compressed(path, X=X_test, y=y_test)
    print(f"saved {path.relative_to(BASE_DIR)}  X={X_test.shape}  y={y_test.shape}")


def save_meta(splits, output_dir: Path, *, partition_mode, client_size_mode, alpha,
              seed, n_clients):
    # meta.json travels with the partition so any run can show how its data looked
    meta = {
        "partition_mode": partition_mode,
        "client_size_mode": client_size_mode,
        "alpha": alpha,
        "seed": seed,
        "n_clients": n_clients,
        "client_sizes": [int(len(y)) for _, y in splits],
        "class_distribution": class_distribution(splits),
    }
    path = output_dir / "meta.json"
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"saved {path.relative_to(BASE_DIR)}")
    return meta

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FL Data Preparation – UCI HAR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n_clients", type=int, default=4,
                        help="Number of FL clients (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--partition-mode", choices=["iid", "label_skew"],
                        default="iid",
                        help="iid = uniform classes per client; "
                             "label_skew = Dirichlet non-IID (default: iid)")
    parser.add_argument("--client-size-mode", choices=["balanced", "imbalanced"],
                        default="balanced",
                        help="balanced = equal client sizes; "
                             "imbalanced = skewed client sizes (default: balanced)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet alpha for label_skew. "
                             "Small=strong non-IID, large=near IID (default: 0.5)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIR,
                        help="Where to write client/test/meta files "
                             "(default: outputs/har).")
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 64)
    print("  UCI HAR – Federated Learning: Data Preparation")
    print("═" * 64 + "\n")

    #load
    print("Download + Load")
    X_train, y_train, X_test, y_test = load_har_real()


    # partition
    print()
    print("Partition of official train split")
    splits = make_partition(
        X_train, y_train, args.n_clients,
        args.partition_mode, args.client_size_mode, args.alpha, args.seed,
    )

    # Save
    print()
    print(f"Saving to {output_dir}")
    save_clients(splits, output_dir)
    save_test_set(X_test, y_test, output_dir)
    save_meta(
        splits, output_dir,
        partition_mode=args.partition_mode,
        client_size_mode=args.client_size_mode,
        alpha=args.alpha,
        seed=args.seed,
        n_clients=args.n_clients,
    )

if __name__ == "__main__":
    main()
