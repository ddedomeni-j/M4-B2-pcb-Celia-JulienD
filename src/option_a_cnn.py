"""Option A — CNN from scratch (PyTorch).

À compléter si votre binôme choisit l'option A.
Architecture suggérée : 3-5 conv layers + 2 dense.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.load_data import CLASSES


class SimpleCNN(nn.Module):
    """CNN simple from scratch pour PCB defect detection.

    Input : (B, 1, 64, 64) niveaux de gris.
    Output : (B, len(CLASSES)) logits.
    """

    def __init__(self, n_classes: int = len(CLASSES)):
        super().__init__()
        # TODO — compléter l'architecture
        # Exemple :
        # self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # ...
        # self.fc1 = nn.Linear(?, 128)
        # self.fc2 = nn.Linear(128, n_classes)
        raise NotImplementedError("TODO — implémenter l'architecture CNN")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO — forward pass
        raise NotImplementedError("TODO — implémenter forward")


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Boucle d'entraînement d'1 epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Évaluation sur un loader (val ou test)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)
    return total_loss / total, correct / total
