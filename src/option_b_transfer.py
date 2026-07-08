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

from src.load_data import CLASSES


def get_transfer_transforms(image_size: int = 224):
    """Transforms pour ResNet — resize 224×224, 3 canaux, normalisation ImageNet.

    Fourni : ce n'est pas l'objet de l'exercice.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),  # 1→3 canaux
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_resnet18_classifier(n_classes: int = len(CLASSES), freeze_backbone: bool = True):
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
    # TODO — implémenter le transfer learning
    #        (cf. ressources/02_Transfer_learning_essentiel.md)
    raise NotImplementedError("TODO — construire le ResNet-18 + nouvelle tête")


# Pour l'entraînement / l'évaluation, réutilise les boucles `train_one_epoch`
# et `evaluate` que tu écris dans src/option_a_cnn.py (même logique).