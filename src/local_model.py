import torch
from torch import nn


class ActivityMLP(nn.Module):
    # ActivityMLP ist unser lokales neuronales Netz fuer die UCI-HAR-Aufgabe.


    # MLP steht fuer "Multi-Layer Perceptron". Das ist ein klassisches
    # Feedforward-Netz: Die Daten fliessen von vorne nach hinten durch mehrere
    # voll verbundene Schichten.
    



    # Wie wird das MLP hier umgesetzt?
    # Der Datensatz ist tabellarisch vorbereitet: Jedes Sample ist ein flacher
    # Vektor mit 561 Features. Wir haben also keine Bilder, Texte oder echten
    # Zeitreihen mehr, sondern bereits berechnete numerische Merkmale. Fuer so
    # eine Eingabe ist ein kleines MLP ein sinnvoller erster Deep-Learning-
    # Ansatz.


    def __init__(self, input_dim=561, hidden_dim=128, num_classes=6, dropout=0.2):


        # nn.Module ist die PyTorch-Basisklasse fuer Modelle.
        # super().__init__() initialisiert die interne PyTorch-Verwaltung, damit
        # Parameter, Submodule, state_dict(), model.to(device) usw. funktionieren.
        super().__init__()

        # nn.Sequential bedeutet: Fuehre alle Layer in genau dieser Reihenfolge
        # aus. Dadurch bleibt die Modellarchitektur kompakt und gut lesbar.
        self.network = nn.Sequential(
            #   Erste lineare Schicht:
            #  Aus input_dim=561 Eingabefeatures werden hidden_dim=128 gelernte
            #  Zwischenwerte. Eine lineare Schicht berechnet grob:
            #   output = input * weights + bias
            nn.Linear(input_dim, hidden_dim),

            #  eLU ist eine Aktivierungsfunktion. Sie macht negative Werte zu 0
            #  und laesst positive Werte unveraendert. Diese Nichtlinearitaett ist
            # wichtig, damit das Modelll komplexere Muster lernen kann.
            nn.ReLU(),

            # Dropout deaktiviert waehrend des Trainings zufaellig einen Anteil
            # der Neuronen. Das wirkt gegen Overfitting, weil das Modell nicht
            # zu stark von einzelnen internen Merkmalen abhaengig werden soll.
            nn.Dropout(dropout),

            # Zweite lineare Schichtt:
            # Wir reduzieren von hidden_dim auf hidden_dim // 2. Das haelt das
            # Modell kleiner und passt gut zur relativ kleinen Client-Datenmenge.
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Output-Schicht:
            # Fuer jede der 6 Aktivitaetsklassen gibt das Modell einen Logit aus.
            #   ogits sind rohe Scores, keine Wahrscheinlichkeiten. Das ist
             #korrekt, weil nn.CrossEntropyLoss direkt Logits erwartet.
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):



        # forward beschreibt den Datenfluss durch das Modell.
        # Wenn im Training model(X) aufgerufen wird, ruft PyTorch intern diese
        # Methode auf.
        return self.network(x)
