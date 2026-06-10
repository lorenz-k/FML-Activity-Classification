from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from src.local_model import ActivityMLP


BASE_DIR = Path(__file__).resolve().parent.parent
FEDERATED_MODEL_DIR = BASE_DIR / "outputs" / "models"
FEDERATED_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_model_parameters(model: ActivityMLP) -> list[np.ndarray]:
    """Return model weights in the list format expected by Flower."""
    return [value.cpu().numpy() for _, value in model.state_dict().items()]


def set_model_parameters(model: ActivityMLP, parameters: list[np.ndarray]) -> None:
    """Load Flower parameters into a PyTorch model."""
    parameter_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        (key, torch.tensor(value)) for key, value in parameter_dict
    )
    model.load_state_dict(state_dict, strict=True)


def save_model_parameters(
    parameters: list[np.ndarray],
    path: Path,
    hidden_dim: int = 128,
    dropout: float = 0.2,
) -> None:
    """Persist Flower parameters as a normal PyTorch state_dict."""
    model = ActivityMLP(hidden_dim=hidden_dim, dropout=dropout)
    set_model_parameters(model, parameters)
    torch.save(model.state_dict(), path)
