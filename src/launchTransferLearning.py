from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.load_data import CLASSES, get_dataloaders
from src.option_b_transfer import (
	build_resnet_classifier,
	evaluate,
	get_transfer_transforms,
	train_one_epoch,
)

def main(data_root=Path("data/pcb_defect_sample")) -> None:
	# data_root = Path("data/pcb_defect_sample")
	model_base_name = "resnet18"
	# model_base_name = "resnet50"
	freeze_backbone = False
	freeze_tag = f"freeze_backbone_{str(freeze_backbone).lower()}"
	output_dir = Path("models") / "transfer" / f"{model_base_name}_{freeze_tag}_{data_root.name}"
	batch_size = 32
	epochs = 5
	lr = 1e-3
	seed = 42
	output_dir.mkdir(parents=True, exist_ok=True)

	torch.manual_seed(seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train_loader, val_loader, test_loader = get_dataloaders(
		root=data_root,
		batch_size=batch_size,
		val_split=0.15,
		test_split=0.15,
		seed=seed,
	)

	# Les 3 Subset partagent le meme dataset de base : on remplace son transform
	# pour produire des tenseurs 3 canaux normalises compatibles ResNet.
	base_dataset = train_loader.dataset.dataset
	base_dataset.transform = get_transfer_transforms()

	model = build_resnet_classifier(
		model_base_name=model_base_name,
		n_classes=len(CLASSES),
		freeze_backbone=freeze_backbone,
	)
	model = model.to(device)

	criterion = nn.CrossEntropyLoss()
	trainable_params = [p for p in model.parameters() if p.requires_grad]
	optimizer = optim.Adam(trainable_params, lr=lr)

	best_val_acc = 0.0
	history = {
		"epoch": [],
		"train_loss": [],
		"train_acc": [],
		"test_loss": [],
		"test_acc": [],
	}

	for epoch in range(1, epochs + 1):
		train_loss, train_acc = train_one_epoch(train_loader, optimizer, criterion, device)
		val_loss, val_acc = evaluate(model, val_loader, criterion, device)
		test_loss, test_acc = evaluate(model, test_loader, criterion, device)

		history["epoch"].append(epoch)
		history["train_loss"].append(train_loss)
		history["train_acc"].append(train_acc)
		history["test_loss"].append(test_loss)
		history["test_acc"].append(test_acc)

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			checkpoint = {
				"epoch": epoch,
				"model_base_name": model_base_name,
				"freeze_backbone": freeze_backbone,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"val_loss": val_loss,
				"val_acc": val_acc,
			}
			improvement_path = output_dir / f"best_{model_base_name}_epoch{epoch:03d}_valacc{val_acc:.4f}.pth"
			latest_best_path = output_dir / "best_model.pth"
			torch.save(checkpoint, improvement_path)
			torch.save(checkpoint, latest_best_path)

		print(
			f"Epoch {epoch}/{epochs} | "
			f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
			f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
			f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
		)

	final_test_loss, final_test_acc = history["test_loss"][-1], history["test_acc"][-1]

	fig, axes = plt.subplots(1, 2, figsize=(12, 5))
	axes[0].plot(history["epoch"], history["train_loss"], label="Train loss", marker="o")
	axes[0].plot(history["epoch"], history["test_loss"], label="Test loss", marker="o")
	axes[0].set_title("Loss")
	axes[0].set_xlabel("Epoch")
	axes[0].set_ylabel("Loss")
	axes[0].grid(True, alpha=0.3)
	axes[0].legend()

	axes[1].plot(history["epoch"], history["train_acc"], label="Train acc", marker="o")
	axes[1].plot(history["epoch"], history["test_acc"], label="Test acc", marker="o")
	axes[1].set_title("Accuracy")
	axes[1].set_xlabel("Epoch")
	axes[1].set_ylabel("Accuracy")
	axes[1].grid(True, alpha=0.3)
	axes[1].legend()

	fig.tight_layout()
	metrics_plot_path = output_dir / "loss_accuracy_train_test.png"
	fig.savefig(metrics_plot_path, dpi=150)
	plt.close(fig)

	print(f"Best val_acc={best_val_acc:.4f}")
	print(f"Final test loss={final_test_loss:.4f} | Final test acc={final_test_acc:.4f}")
	print(f"Checkpoints enregistrés dans: {output_dir}")
	print(f"Graphique enregistré: {metrics_plot_path}")


if __name__ == "__main__":
	data_root = Path("data/pcb_defect_sample_1000")
	main(data_root)
