"""Generate a synthetic PCB Defect dataset for the M4-B2 brief.

Reproducible (``random_state=42``). 2 100 images 64×64 niveaux de gris,
réparties en 7 classes (1 OK + 6 défauts) — 300 images par classe.

Les classes reproduisent la taxonomie du dataset PCB Defect Detection
réel (open / short / mousebite / spur / copper / pin-hole + OK), mais
visuellement stylisées (pistes claires sur fond sombre).

Design pédagogique (M4-B2) : le problème doit être **apprenable** dans le
budget CPU du brief (CNN scratch ~60-85 %, transfer plus haut). Pour cela :
- un **fond PCB fixe** (template commun à toutes les images) → faible
  variance intra-classe ;
- la variation vient d'une **augmentation légère** (bruit + luminosité),
  pas d'une régénération totale du fond ;
- chaque défaut a une **signature nette** (forme + contraste + taille
  20-40 px) → le signal de classe domine.

Run from this folder::

    python generate_dataset.py
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RANDOM_STATE: int = 42
N_PER_CLASS: int = 300
IMG_SIZE: int = 64
OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "pcb_defect_sample"

CLASSES: list[str] = [
    "ok",
    "open",       # circuit ouvert (piste coupée)
    "short",      # court-circuit (pont clair entre pistes)
    "mousebite",  # mordillure sur bord
    "spur",       # protubérance
    "copper",     # excès de cuivre (grosse tache)
    "pin_hole",   # trous
]


def build_fixed_base() -> Image.Image:
    """Construit le **template PCB fixe** (identique pour toutes les images).

    Fond commun → la seule chose qui change entre classes est le défaut.
    Déterministe (aucun aléa) : c'est ce qui rend le problème apprenable.
    """
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=25)  # fond sombre
    draw = ImageDraw.Draw(img)

    # Pistes verticales + horizontales à positions FIXES
    for x in (14, 30, 46):
        draw.line([(x, 4), (x, IMG_SIZE - 4)], fill=170, width=2)
    for y in (20, 44):
        draw.line([(4, y), (IMG_SIZE - 4, y)], fill=170, width=2)

    # Pads (cercles pleins) à positions FIXES
    for (x, y) in ((14, 20), (46, 44), (30, 32)):
        draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], fill=210)

    return img


def apply_defect(img: Image.Image, defect: str, rng: random.Random) -> Image.Image:
    """Applique un défaut **grand et contrasté** selon la classe.

    Position légèrement variable (jitter modéré) mais forme/taille/contraste
    caractéristiques → signal de classe fort.
    """
    if defect == "ok":
        return img  # base propre

    draw = ImageDraw.Draw(img)
    # centre du défaut : position variable sur la carte (jitter large) → le
    # modèle doit reconnaître la FORME, pas mémoriser une position.
    cx = rng.randint(16, IMG_SIZE - 16)
    cy = rng.randint(16, IMG_SIZE - 16)

    if defect == "open":
        # bloc SOMBRE (piste coupée) — rectangle vertical
        draw.rectangle([(cx - 3, cy - 8), (cx + 3, cy + 8)], fill=40)
    elif defect == "short":
        # pont CLAIR reliant deux pistes — barre diagonale
        draw.line([(cx - 9, cy - 9), (cx + 9, cy + 9)], fill=220, width=3)
    elif defect == "mousebite":
        # encoche SOMBRE demi-circulaire sur un bord
        edge_x = rng.choice([0, IMG_SIZE - 1])
        draw.pieslice([(edge_x - 8, cy - 8), (edge_x + 8, cy + 8)], 0, 360, fill=40)
    elif defect == "spur":
        # protubérance CLAIRE en triangle (pointe qui dépasse)
        draw.polygon(
            [(cx - 7, cy + 5), (cx + 7, cy + 5), (cx, cy - 10)], fill=220
        )
    elif defect == "copper":
        # tache CLAIRE informe (excès de cuivre)
        draw.ellipse([(cx - 9, cy - 6), (cx + 9, cy + 6)], fill=215)
    elif defect == "pin_hole":
        # plusieurs trous SOMBRES nets (motif caractéristique)
        for dx, dy in ((-6, -6), (6, -6), (0, 6)):
            draw.ellipse([(cx + dx - 2, cy + dy - 2), (cx + dx + 2, cy + dy + 2)], fill=40)

    return img


def augment(img: Image.Image, rng_np: np.random.Generator) -> Image.Image:
    """Variation intra-classe **légère** (bruit + luminosité), pas de régénération."""
    arr = np.array(img, dtype=np.float32)
    arr *= rng_np.uniform(0.8, 1.2)              # jitter luminosité
    arr += rng_np.normal(0, 12, arr.shape)       # bruit gaussien modéré
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def main() -> None:
    """Generate the dataset structured in folders per class."""
    rng_random = random.Random(RANDOM_STATE)
    rng_np = np.random.default_rng(seed=RANDOM_STATE)

    base = build_fixed_base()  # template commun (fixe)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        (OUTPUT_DIR / cls).mkdir(parents=True, exist_ok=True)

    total = 0
    for cls in CLASSES:
        print(f"Génération classe {cls} ({N_PER_CLASS} images)...")
        for i in range(N_PER_CLASS):
            img = base.copy()
            img = apply_defect(img, cls, rng_random)
            img = augment(img, rng_np)
            img.save(OUTPUT_DIR / cls / f"{cls}_{i:04d}.png")
            total += 1

    print(f"\n✓ {total:,} images générées dans {OUTPUT_DIR}")
    print(f"  Structure : {OUTPUT_DIR.name}/<classe>/<classe>_NNNN.png")


if __name__ == "__main__":
    main()
