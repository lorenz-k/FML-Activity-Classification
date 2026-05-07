"""
Federated Learning UCI HAR Data Preparation
==============================================

Outputs (./outputs/har/):
  client_0.npz … client_3.npz  

Pipeline:
  1. merge data into one set
  2. Partition random IID split into N equal-ish client shards

Usage:
  python prepare_har.py                       # 4 clients
  python prepare_har.py --n_clients 4 --seed 42
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

# paths
BASE_DIR   = Path(__file__).parent.parent
CACHE_DIR  = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "outputs" / "har"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def load_har_real():
    #Download, extract, merge and return raw X, Y.
    download_har()
    extract_har()

    X_tr, y_tr = load_split("train")
    X_te, y_te = load_split("test")

    X = np.vstack([X_tr, X_te])     # (10299, 561)
    y = np.concatenate([y_tr, y_te])    # (10299,)
    print(f"merged data. total: X={X.shape}  y={y.shape}")
    return X, y



# IID partition

def partition_iid(X: np.ndarray, y: np.ndarray, n_clients: int, seed: int):
    # partition into n_clients partittions with random shuffle and random inedx split
    idx = np.random.default_rng(seed).permutation(len(y))
    shards = np.array_split(idx, n_clients)
    splits = [(X[s], y[s]) for s in shards]
    print(f"split data into {n_clients} clients, "
          f"sizes = {[len(s[1]) for s in splits]}")
    return splits


# save

def save_clients(splits):
    # write client_i.npz files

    for i, (X, y) in enumerate(splits):
        path = OUTPUT_DIR / f"client_{i}.npz"
        np.savez_compressed(path, X=X, y=y)

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
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "═" * 64)
    print("  UCI HAR – Federated Learning: Data Preparation")
    print("═" * 64 + "\n")

    #load
    print("Download + Load")
    X, y = load_har_real()


    # partition
    print()
    print("IID Partition")
    splits = partition_iid(X, y, args.n_clients, args.seed)

    # Save
    print()
    print("Saving")
    save_clients(splits)

if __name__ == "__main__":
    main()