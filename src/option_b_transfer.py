"""Option B — Transfer learning (ResNet-18 pré-entraîné).

À IMPLÉMENTER si votre binôme choisit l'option B.
Stratégie : freeze backbone + fine-tune classifier head.
Mini-cours d'appui : ressources/02_Transfer_learning_essentiel.md

Note : ResNet attend des images **3 canaux**. Si vos PNG sont en niveaux de
gris (1 canal), répliquez le canal x3 dans les transforms (déjà géré ci-dessous).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models, transforms


def build_resnet18_classifier(
    n_classes: int = 7, freeze_backbone: bool = True
):
    """Construit un ResNet-18 pré-entraîné avec une nouvelle tête de classification.

    À faire (cf. mini-cours 02) :
    1. charger `models.resnet18` avec les poids pré-entraînés ImageNet
    2. si `freeze_backbone`, geler tous les paramètres du backbone
    3. remplacer la dernière couche `model.fc` par une `nn.Linear` vers `n_classes`

    Args:
        n_classes: nombre de classes finales.
        freeze_backbone: si True, seule la tête de classification est fine-tunée.

    Returns:
        nn.Module prêt à l'entraînement.
    """
    # 1. charger `models.resnet18` avec les poids pré-entraînés ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        # 2. geler tous les paramètres du backbone pour garder les couches pré-entraînées
        for param in model.parameters():
            param.requires_grad = False

    # 3. remplacer la dernière couche `model.fc` par une `nn.Linear` vers `n_classes`
    model.fc = nn.Linear(
        model.fc.in_features, n_classes
    )  

    return model


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


# Pour l'entraînement / l'évaluation, réutilise les boucles `train_one_epoch`
# et `evaluate` que tu écris dans src/option_a_cnn.py (même logique).
