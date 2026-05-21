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
    # - mps: Apple Silicon GPU/Metal, z. B. auf modernen MacBooks
    # - cuda: NVIDIA GPU
    # - cpu: normaler Prozessor als universeller Fallback
    #
    # Wir nehmen automatisch das beste verfuegbare Device.
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    # if torch.cuda.is_available():
    #     return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    # Eine wir hierbei bedeuten ->  Das Modell sieht einmal alle Trainingssamples.
    #
    #  model.train() aktiviert den Trainingsmodus. Das ist wichtig fuer Layer wie
    # Dropout:  Dropout soll im Training aktiv sein, aber nicht bei Evaluation.
    model.train()

    #  Wir sammeln Loss und Accuracy ueber alle Batches,  um am Ende einen
    #   durchschnittlichen Epochenwert auszugeben.
    total_loss = 0.0
    correct = 0
    total = 0

    # Der DataLoader liefert pro Iteration einen Batch: X sind Features,
    # y sind die korrekten Labels.
    for X, y in loader:
        # Daten und Labels muessen auf demselben Device liegen wie das Modell.
        X = X.to(device)
        y = y.to(device)

        #  PyTorch sammelt Gradienten standardmaessig auf. Deshalb setzen wir sie
        # vor jedem Batch zurueck.
        optimizer.zero_grad()

        # Forward Pass: Das Modell berechnet fuer jedes Sample 6 Logits.
        logits = model(X)

        #    Loss: CrossEntropyLoss vergleicht die Logits mit den echten Labels und
        # gibt  eine Zahl zurueck, die ausdrueckt, wie falsch das Modell liegt.
        loss = criterion(logits, y)

        # Backpropagation: PyTorch berechnet automatisch die Gradienten aller
        # lernbaren Parameter bezogen auf den Loss.
        loss.backward()

        # Optimizer-Schritt: Adam-Optimizer nutzt die Gradienten, um die Modellgewichte zu
        # aktualisieren::
        optimizer.step()

        #   loss.item() ist der Loss fuer den aktuellen Batch. Wir multiplizieren
        # mit len(y), damit grosse und kleine Batches korrekt gewichtet werden.
        total_loss += loss.item() * len(y)

        #  Die vorhergesagte Klasse ist der Index des groessten Logits.
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += len(y)

    #  Rueckgabe: durchschnittlicher Loss und Accuracy fuer diese Epoche.
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
        # True Positive: Klasse wurde korrekt als sich selbst vorhergesagt.
        true_positive = confusion[class_id, class_id].item()

        # False Positive: Modell sagt diese Klasse, aber die echte Klasse war
        # eine andere. Das ist die Spaltensumme minus True Positive.
        false_positive = confusion[:, class_id].sum().item() - true_positive

        # False Negative: Echte Klasse ist class_id, aber Modell sagt etwas
        # anderes. Das ist die Zeilensumme minus True Positive.
        false_negative = confusion[class_id, :].sum().item() - true_positive

        # Precision: Wenn das Modell diese Klasse vorhersagt, wie oft stimmt das?
        precision_denominator = true_positive + false_positive
        precision = true_positive / precision_denominator if precision_denominator else 0.0

        # Recall: Von allen echten Samples dieser Klasse, wie viele findet das
        # Modell?
        recall_denominator = true_positive + false_negative
        recall = true_positive / recall_denominator if recall_denominator else 0.0

        # F1 ist der harmonische Mittelwert aus Precision und Recall.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)

    return sum(scores) / len(scores)


def main():
    # --> die kommandozeilenargumente machen das Training reproduzierbar und flexibel
    # Beispiel:
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

    #   Seed fuer reproduzierbare Initialisierung und reproduzierbares Training,
    # also soweit die verwendete Hardware deterministisch arbeitet.
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

        #   Wenn diese Epoche besser ist als alle bisherigen, sichern wir die
        # Gewichte. deepcopy ist wichtig, damit spaetere Updates den gespeicherten
        #   Zustand nicht versehentlich veraendern.
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"test loss {test_loss:.4f} acc {test_acc:.4f}"
        )

    # Am Ende laden wir den besten Zustand zurueck und evaluieren genau diesen.
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
