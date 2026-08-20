import time
import torch
from PIL import Image
from test_transfer_confusion import load_model_from_checkpoint
from test_transfer_confusion import MODEL_CONFIGS
from load_data import get_default_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement de l'image
image = Image.open("./data/pcb_defect_sample/copper/copper_0000.png").convert("RGB")
transfrom = get_default_transforms()
input_tensor = transfrom(image).unsqueeze(0).to(device)

for model_base_name, freeze_backbone in MODEL_CONFIGS:
    # Chargement du modèle
    model, checkpoint_path = load_model_from_checkpoint(
        model_base_name=model_base_name,
        freeze_backbone=freeze_backbone,
        device=device,
    )

    # Warm-up (important surtout sur GPU)
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # Synchronisation GPU si nécessaire
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Mesure
    start = time.perf_counter()

    with torch.no_grad():
        outputs = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print(f"Temps d'inférence {model_base_name} freeze_backbone {freeze_backbone} : {elapsed*1000:.2f} ms")