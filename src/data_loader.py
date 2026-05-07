import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

def get_client_loader(client_id, batch_size=32):
    # datei als numpy array laden 
    data = np.load(f"../outputs/har/client_{client_id}.npz")
    
    # in pytorch tensoren umwandeln
    X = torch.tensor(data['X'], dtype=torch.float32)

    #long anscheinend, dass loss funktion mit arbeiten kann
    y = torch.tensor(data['y'], dtype=torch.long)
    
    # Dataset & Loader erstellen
    dataset = TensorDataset(X, y)


    #wir laden immer nur batches in der grösse von (standardmäßig) 32 Zeilen in das Modell (nicht alle gleichzeitig)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)