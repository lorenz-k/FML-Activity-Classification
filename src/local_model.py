import torch
from torch import nn


class ActivityMLP(nn.Module):
    # ActivityMLP ist unser lokales neuronales Netz fuer die UCI-HAR-Aufgabe.


    def __init__(self, input_dim=561, hidden_dim=128, num_classes=6, dropout=0.2):


        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):



        return self.network(x)
