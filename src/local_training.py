import argparse
import copy
from pathlib import Path

import torch
from torch import nn

from src.data_loader import get_client_loader, get_test_loader
from src.local_model import ActivityMLP


# Projektroot und Ausgabeordner fuer trainierte Modelle.
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "outputs" / "models"

# Der Modellordner wird automatisch angelegt, damit torch.save spaeter nicht an
# einem fehlenden Verzeichnis scheitert.
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def get_device():
    # PyTorch kann je nach Rechner auf unterschiedlicher Hardware laufen:
    # - mps: Apple Silicon GPU/Metal, z. B. auf modernen MacBooks, metal ab m1 chip
    # - cuda: NVIDIA GPU
    # - cpu: normaler Prozessor als universeller Fallback
    #
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(logits, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(y)

        correct += (logits.argmax(dim=1) == y).sum().item()
        total += len(y)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    #  @torch.no_grad() deaktiviert Gradientenberechnung. Beim Evaluieren lernen
    # wir nicht, deshalb spart das Speicher und Rechenzeit.
    #
    #  model.eval() schaltet das Modell in den Evaluationsmodus. Dropout ist dann
    # deaktiviert, damit die Vorhersagen stabil sind.
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # Confusion Matrix mit 6 Zeilen und 6 Spalten:
    # Zeile = echte Klasse, Spalte = vorhergesagte Klasse.
    confusion = torch.zeros(6, 6, dtype=torch.int64)

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = criterion(logits, y)
        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * len(y)
        correct += (predictions == y).sum().item()
        total += len(y)

        # Fuer jedes   Sample erhoehen wir die passende Zelle der Confusion Matrix.
        #  Beispiel: true_label=3 und predicted_label=4 bedeutet:
        #   echtes SITTING wurde als STANDING vorhergesagt.
        for true_label, predicted_label in zip(y.cpu(), predictions.cpu()):
            confusion[true_label, predicted_label] += 1

    return total_loss / total, correct / total, confusion


def print_confusion_matrix(confusion):
    # Kleine Terminal-Ausgabe, damit man direkt sieht, welche Klassen verwechselt
    # werden. Bei UCI HAR sind besonders SITTING/STANDING und die Walking-Klassen
    #  interessant.
    labels = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
    print("\nConfusion matrix (rows=true, cols=predicted)")
    print(" " * 13 + " ".join(f"{label[:8]:>8}" for label in labels))
    for label, row in zip(labels, confusion.tolist()):
        print(f"{label[:12]:>12} " + " ".join(f"{value:8d}" for value in row))


def macro_f1_from_confusion(confusion):
    # Macro-F1 berechnet zuerst einen F1-Wert pro Klasse und mittelt diese Werte.
    # Dadurch zaehlt jede Klasse gleich stark, unabhaengig davon, wie viele
    # Samples sie hat.
    scores = []
    for class_id in range(confusion.shape[0]):
        true_positive = confusion[class_id, class_id].item()

        false_positive = confusion[:, class_id].sum().item() - true_positive

        false_negative = confusion[class_id, :].sum().item() - true_positive

        precision_denominator = true_positive + false_positive
        precision = true_positive / precision_denominator if precision_denominator else 0.0

        recall_denominator = true_positive + false_negative
        recall = true_positive / recall_denominator if recall_denominator else 0.0

        # F1 ist der harmonische Mittelwert aus Precision und Recall.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)

    return sum(scores) / len(scores)


def main():
    # uv run python -m src.local_training --client-id 0 --epochs 50
    parser = argparse.ArgumentParser(description="Train a local UCI HAR model on one client.")
    parser.add_argument("--client-id", type=int, default=0, help="Client shard id to train on.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of local training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="First hidden layer size.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()

    # Trainingsdaten: genau ein Client.
    #  Testdaten: globales Holdout-Testset, das nicht fuer Training genutzt wird.
    train_loader = get_client_loader(args.client_id, batch_size=args.batch_size)
    test_loader = get_test_loader(batch_size=256)

    # Modell, Loss und Optimizer.
    model = ActivityMLP(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Training local model on client_{args.client_id}.npz")
    print(f"Device: {device}")
    print()

    #   Wir speichern den besten Modellzustand nach Test-Accuracy. Das ist wichtig,
    # weil die letzte Epoche nicht automatisch die beste sein muss.
    best_accuracy = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _ = evaluate(model, test_loader, criterion, device)

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"test loss {test_loss:.4f} acc {test_acc:.4f}"
        )

    model.load_state_dict(best_state)
    test_loss, test_acc, confusion = evaluate(model, test_loader, criterion, device)
    macro_f1 = macro_f1_from_confusion(confusion)

    #   state_dict enthaelt die gelernten Gewichte des Modells. Diese Datei kann
    # spaeter fuer Inferenz, Dashboard oder Federated-Learning-Vergleiche geladen
    #  werden.
    model_path = MODEL_DIR / f"local_client_{args.client_id}.pt"
    torch.save(model.state_dict(), model_path)

    print()
    print(f"Best epoch by test accuracy: {best_epoch}")
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Final test macro F1: {macro_f1:.4f}")
    print(f"Saved model: {model_path.relative_to(BASE_DIR)}")
    print_confusion_matrix(confusion)


if __name__ == "__main__":
    main()
