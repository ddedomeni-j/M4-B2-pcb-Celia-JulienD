"""Option C — Zero-shot avec foundation model (CLIP).

À IMPLÉMENTER si votre binôme choisit l'option C.
Pas d'entraînement — juste de l'inférence avec prompts par classe.
Mini-cours d'appui : ressources/03_Zero_shot_CLIP_essentiel.md

CLIP `clip-vit-base-patch32` HuggingFace, ~150 Mo, CPU OK (~80-200 ms/image).
"""
from __future__ import annotations

from pathlib import Path
import time 

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from load_data import CLASSES
from collections import defaultdict

import cv2

MODEL_ID: str = "openai/clip-vit-base-patch32"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO — écris UNE description textuelle par classe (cf. mini-cours 03).
# La qualité de ces prompts fait TOUTE la précision du zero-shot : c'est le
# cœur de l'option C. Une entrée par classe de CLASSES (les 7).
# Exemple de format (à compléter et raffiner) :
CLASS_PROMPTS: dict[str, str] = {
    "ok": "a close-up image of a printed circuit board with regular traces, clean geometry, symmetric patterns and no visible defect. Image has 3 perfect vertical lines, 3 perfect horizontal lines and 3 plain disks with no holes",
    "copper":"a photo of a printed circuit board with a copper defect. The image has n ellipse on top of the correct circuit",
    "pin_hole":"a printed circuit board with a missing hole defect. One of the 3 disk or line is cropped by a disk",
    "mousebite":"a printed circuit board with a mouse bite defect. One of the straight line is cropped",
    "spur":"a printed circuit board with a spur defect. This image contains a triangle",
    "short":"a printed circuit board with a short circuit defect. There is an asymetric line on the image cutting horizontal and vertical line",
    "open":"a printed circuit board with an open circuit defect. 3 vertical lines, 3 horizontal lines and at least one of the lines has a hole",
}

def load_clip_model():
    """Charge CLIP processor + model (mise en cache locale au 1ᵉʳ appel). Fourni."""

    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID)
    model.to(DEVICE)
    model.eval()

    return processor, model


def classify_image(image_path: Path, processor, model, prompt) -> str:
    """Classifie une image via CLIP zero-shot, retourne la classe prédite.

    À faire (cf. mini-cours 03) :
      1. ouvrir l'image en RGB
      2. construire la liste des prompts depuis CLASS_PROMPTS (ordre de CLASSES)
      3. passer (text=prompts, images=image) au processor puis au model
      4. softmax sur `logits_per_image`, retourner la classe `argmax`
    """
    # TODO — inférence zero-shot CLIP
    # image = Image.open(image_path).convert("RGB")
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.imshow("image", cv2.resize(image, None, fx = 4, fy = 4))
    cv2.waitKey()

    texts = list(prompt.values())

    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    start = time.process_time()
    with torch.no_grad():
        outputs = model(**inputs)
    end = time.process_time()
    print("CPU Time:", (end - start) * 1e3, "ms")
    
    logits = outputs.logits_per_image
    probs = logits.softmax(dim=1)

    for cls, prob in zip(prompt.keys(), logits.softmax(dim=1)[0].cpu().numpy()):
        print(f"{cls:15s} : {prob:.4f}")

    print("Prediction :", list(prompt.keys())[probs.argmax().item()])
    print("-" * 50)


    best_idx = probs.argmax().item()

    return list(prompt.keys())[best_idx]


def evaluate_zero_shot(image_dir: Path, processor, model, prompt=CLASS_PROMPTS, max_samples: int | None = None):
    """Évalue CLIP zero-shot sur le dataset PCB.

    À faire : parcourir chaque classe, prédire via `classify_image`, agréger.

    Returns:
        dict {classe: [correct, total]}.
    """
    # TODO — boucle d'évaluation
    
    results = defaultdict(lambda: [0, 0])

    for true_class in CLASSES:
        class_dir = image_dir / true_class

        if not class_dir.exists():
            print(f"[WARNING] Dossier absent : {class_dir}")
            continue

        image_paths = []

        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_paths.extend(class_dir.glob(ext))

        image_paths = sorted(image_paths)

        if max_samples is not None:
            image_paths = image_paths[:max_samples]

        print(
            f"Évaluation de {true_class} "
            f"({len(image_paths)} images)"
        )

        for image_path in image_paths:

            pred_class = classify_image(
                image_path,
                processor,
                model,
                prompt
            )

            results[true_class][1] += 1  # total

            if pred_class == true_class:
                results[true_class][0] += 1  # correct

    return dict(results)
