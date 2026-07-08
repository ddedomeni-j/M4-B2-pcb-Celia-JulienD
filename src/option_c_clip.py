"""Option C — Zero-shot avec foundation model (CLIP).

À IMPLÉMENTER si votre binôme choisit l'option C.
Pas d'entraînement — juste de l'inférence avec prompts par classe.
Mini-cours d'appui : ressources/03_Zero_shot_CLIP_essentiel.md

CLIP `clip-vit-base-patch32` HuggingFace, ~150 Mo, CPU OK (~80-200 ms/image).
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.load_data import CLASSES

MODEL_ID: str = "openai/clip-vit-base-patch32"

# TODO — écris UNE description textuelle par classe (cf. mini-cours 03).
# La qualité de ces prompts fait TOUTE la précision du zero-shot : c'est le
# cœur de l'option C. Une entrée par classe de CLASSES (les 7).
# Exemple de format (à compléter et raffiner) :
CLASS_PROMPTS: dict[str, str] = {
    # "ok": "a photograph of a clean PCB with no defects",
    # ... à compléter pour les 7 classes de CLASSES
}


def load_clip_model():
    """Charge CLIP processor + model (mise en cache locale au 1ᵉʳ appel). Fourni."""
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


def classify_image(image_path: Path, processor, model) -> str:
    """Classifie une image via CLIP zero-shot, retourne la classe prédite.

    À faire (cf. mini-cours 03) :
      1. ouvrir l'image en RGB
      2. construire la liste des prompts depuis CLASS_PROMPTS (ordre de CLASSES)
      3. passer (text=prompts, images=image) au processor puis au model
      4. softmax sur `logits_per_image`, retourner la classe `argmax`
    """
    # TODO — inférence zero-shot CLIP
    raise NotImplementedError("TODO — classifier l'image via CLIP")


def evaluate_zero_shot(image_dir: Path, processor, model, max_samples: int | None = None):
    """Évalue CLIP zero-shot sur le dataset PCB.

    À faire : parcourir chaque classe, prédire via `classify_image`, agréger.

    Returns:
        dict {classe: [correct, total]}.
    """
    # TODO — boucle d'évaluation
    raise NotImplementedError("TODO — évaluer le zero-shot sur le dataset")