
import os
from pathlib import Path


import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


#Base directory
BASE_DIR = Path(os.environ.get("FML_BASE_DIR", Path(__file__).resolve().parent.parent))


#output dir with har files
OUTPUT_DIR = BASE_DIR / "outputs" / "har"


def load_npz_dataset(path: Path):
    # datei als numpy arrays laden 
    data = np.load(path)

    # in pytorch tensoren umwandeln --> float32. Das ist der uebliche Datentyp 
    X = torch.tensor(data["X"], dtype=torch.float32)

    # Labels sind indizes 0..5 --> Integer-Tensor --> long
    y = torch.tensor(data["y"], dtype=torch.long)

    # X und y sampleweise koppeln --> dataset[i] --> (X[i], y[i]).
    return TensorDataset(X, y)


def get_loader(path: Path, batch_size=32, shuffle=True):
    # laden -> TensorDataset bauen -> DataLoader erzeugen.
    dataset = load_npz_dataset(path)

    # DataLoader liefert die Daten batchweise. Statt alle Samples auf einmal zu
    # trainieren, bekommt das Modell pro Schritt z. B. 32 oder 64 Samples.
    #
    # shuffle=True --> wir shuffeln
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_client_loader(client_id, batch_size=32):
    #lädt einen client , z.B. client_id=0 -> path=outputs/har/client_0.npz.
    path = OUTPUT_DIR / f"client_{client_id}.npz"
    return get_loader(path, batch_size=batch_size, shuffle=True)


def get_test_loader(batch_size=256):
    # testset laden 

    path = OUTPUT_DIR / "test_global.npz"
    return get_loader(path, batch_size=batch_size, shuffle=False)
