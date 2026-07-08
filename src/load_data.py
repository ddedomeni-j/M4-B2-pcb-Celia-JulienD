"""M4-B2 — Chargement du dataset PCB Defect dans un torch Dataset.

Structure attendue dans `data/pcb_defect_sample/` :
    pcb_defect_sample/
    ├── ok/
    ├── open/
    ├── short/
    ├── mousebite/
    ├── spur/
    ├── copper/
    └── pin_hole/

Chaque dossier contient ~300 images PNG 64×64 niveaux de gris.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASSES: list[str] = [
    "ok", "open", "short", "mousebite", "spur", "copper", "pin_hole",
]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: c for c, i in CLASS_TO_IDX.items()}


class PCBDefectDataset(Dataset):
    """Dataset PyTorch pour les images PCB."""

    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for cls in CLASSES:
            cls_dir = self.root / cls
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.glob("*.png")):
                self.samples.append((img_path, CLASS_TO_IDX[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")  # niveaux de gris
        if self.transform:
            img = self.transform(img)
        return img, label


def get_default_transforms(image_size: int = 64):
    """Transforms par défaut : resize + tensor."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # → [0, 1]
    ])


def get_dataloaders(
    root: Path,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader).

    Split **aléatoire mais reproductible** (generator seedé). ⚠️ `random_split`
    n'est **pas** stratifié — acceptable ici car le dataset est **équilibré**
    (300 images/classe) → les splits sont ≈ équilibrés. Pense aussi à fixer
    `torch.manual_seed(seed)` avant l'entraînement pour que la comparaison
    entre binômes soit reproductible.
    """
    dataset = PCBDefectDataset(root, transform=get_default_transforms())

    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_test = int(n_total * test_split)
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
