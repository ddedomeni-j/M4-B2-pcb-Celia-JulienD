from __future__ import annotations

from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.load_data import CLASSES, get_dataloaders
from src.option_b_transfer import build_resnet_classifier, get_transfer_transforms


MODEL_CONFIGS: list[tuple[str, bool]] = [
    ("resnet18", True),
    ("resnet18", False),
    ("resnet50", True),
    ("resnet50", False),
]


def checkpoint_path_for(
    model_base_name: str,
    freeze_backbone: bool,
    checkpoint_suffix: str = "",
) -> Path:
    suffix = f"_{checkpoint_suffix}" if checkpoint_suffix else ""
    model_dir = (
        PROJECT_ROOT
        / "models"
        / "transfer"
        / f"{model_base_name}_freeze_backbone_{str(freeze_backbone).lower()}{suffix}"
    )
    return model_dir / "best_model.pth"


def load_model_from_checkpoint(
    model_base_name: str,
    freeze_backbone: bool,
    device: torch.device,
    checkpoint_suffix: str = "",
) -> tuple[torch.nn.Module, Path]:
    checkpoint_path = checkpoint_path_for(
        model_base_name=model_base_name,
        freeze_backbone=freeze_backbone,
        checkpoint_suffix=checkpoint_suffix,
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_resnet_classifier(
        model_base_name=model_base_name,
        n_classes=len(CLASSES),
        freeze_backbone=freeze_backbone,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint_path


def predict_all(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch.to(device))
            preds = logits.argmax(dim=1).cpu()
            y_pred.extend(preds.tolist())
            y_true.extend(y_batch.tolist())

    return y_true, y_pred


def plot_confusion_matrices(
    cms: list[tuple[str, float, list[list[int]]]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes_flat = axes.flatten()

    for idx, (title, accuracy, cm) in enumerate(cms):
        ax = axes_flat[idx]
        image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{title}\naccuracy={accuracy:.2%}")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right")
        ax.set_yticklabels(CLASSES)

        threshold = max(max(row) for row in cm) / 2.0 if cm else 0
        for i in range(len(CLASSES)):
            row_total = sum(cm[i])
            for j in range(len(CLASSES)):
                color = "white" if cm[i][j] > threshold else "black"
                pct = (cm[i][j] / row_total * 100.0) if row_total > 0 else 0.0
                ax.text(
                    j,
                    i,
                    f"{cm[i][j]}\n{pct:.1f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(len(cms), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_single_confusion_matrix(
    title: str,
    accuracy: float,
    cm: list[list[int]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"{title}\naccuracy={accuracy:.2%}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)

    threshold = max(max(row) for row in cm) / 2.0 if cm else 0
    for i in range(len(CLASSES)):
        row_total = sum(cm[i])
        for j in range(len(CLASSES)):
            color = "white" if cm[i][j] > threshold else "black"
            pct = (cm[i][j] / row_total * 100.0) if row_total > 0 else 0.0
            ax.text(
                j,
                i,
                f"{cm[i][j]}\n{pct:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace la matrice de confusion (split validation) pour 4 modèles transfer learning."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "single"],
        default="all",
        help="all: 4 matrices; single: une seule matrice ciblée",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "pcb_defect_sample",
        help="Dossier dataset complet (par défaut: data/pcb_defect_sample)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size d'inférence (par défaut: 64)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Proportion validation (par défaut: 0.15)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="Proportion test interne du split (par défaut: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed du split aléatoire (par défaut: 42)",
    )
    parser.add_argument(
        "--model-base-name",
        type=str,
        choices=["resnet18", "resnet50"],
        default="resnet18",
        help="Modèle ciblé en mode single",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Mode single: utilise le checkpoint freeze_backbone=true",
    )
    parser.add_argument(
        "--checkpoint-suffix",
        type=str,
        default="",
        help="Suffixe du dossier checkpoint (ex: pcb_defect_sample_1000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Fichier image de sortie",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = get_dataloaders(
        root=args.dataset_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )

    # Les subsets train/val/test partagent le même dataset de base.
    # On force les transforms compatibles ResNet pour l'inférence transfer.
    base_dataset = train_loader.dataset.dataset
    base_dataset.transform = get_transfer_transforms()

    if len(val_loader.dataset) == 0:
        raise ValueError("Le split validation est vide: ajuste val_split/test_split")

    if args.mode == "single":
        model, checkpoint_path = load_model_from_checkpoint(
            model_base_name=args.model_base_name,
            freeze_backbone=args.freeze_backbone,
            device=device,
            checkpoint_suffix=args.checkpoint_suffix,
        )
        y_true, y_pred = predict_all(model, val_loader, device)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES)))).tolist()
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        accuracy = correct / len(y_true)

        label = f"{args.model_base_name} | freeze_backbone={args.freeze_backbone}"
        if args.checkpoint_suffix:
            label = f"{label} | ckpt_suffix={args.checkpoint_suffix}"

        output_path = args.output
        if output_path is None:
            suffix = f"_{args.checkpoint_suffix}" if args.checkpoint_suffix else ""
            output_path = (
                PROJECT_ROOT
                / "models"
                / "transfer"
                / f"confusion_matrix_{args.model_base_name}_freeze_{str(args.freeze_backbone).lower()}{suffix}_validation.png"
            )

        plot_single_confusion_matrix(label, accuracy, cm, output_path)
        print(f"{label}")
        print(f"checkpoint: {checkpoint_path}")
        print(f"accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"Image enregistrée: {output_path}")
    else:
        results: list[tuple[str, float, list[list[int]]]] = []

        for model_base_name, freeze_backbone in MODEL_CONFIGS:
            model, checkpoint_path = load_model_from_checkpoint(
                model_base_name=model_base_name,
                freeze_backbone=freeze_backbone,
                device=device,
            )

            y_true, y_pred = predict_all(model, val_loader, device)
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES)))).tolist()
            correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
            accuracy = correct / len(y_true)

            label = f"{model_base_name} | freeze_backbone={freeze_backbone}"
            results.append((label, accuracy, cm))

            print(f"{label}")
            print(f"checkpoint: {checkpoint_path}")
            print(f"accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
            print("-" * 60)

        output_path = args.output
        if output_path is None:
            output_path = (
                PROJECT_ROOT
                / "models"
                / "transfer"
                / "confusion_matrices_4_models_validation.png"
            )

        plot_confusion_matrices(results, output_path)
        print(f"Image enregistrée: {output_path}")


if __name__ == "__main__":
    main()
