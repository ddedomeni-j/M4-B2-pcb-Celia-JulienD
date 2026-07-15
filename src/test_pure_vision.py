from pathlib import Path
import numpy as np
import pandas as pd
from pure_vision import *

# image_path = Path("data/pcb_defect_sample/ok/ok_0000.png")
# image_path = Path("data/pcb_defect_sample/open/open_0003.png")
root_dir = Path("data/test")
# root_dir = Path("data/pcb_defect_sample")
rows = []
image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
for class_dir in sorted(root_dir.iterdir()):
    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    image_paths = sorted(
        p for p in class_dir.rglob("*")
        if p.suffix.lower() in image_extensions
    )

    print(f"Classe {class_name} : {len(image_paths)} images")

    for image_path in image_paths:
        try:
            print(f"Image : {image_path.name}")

            # On seuille pour avoir du blanc sur noir
            gray, binary = load_binary_image(image_path)

            # on détecte les lignes 
            lines = detect_lines(binary)
            line_summary = count_lines(lines)

            # print("Line summary:")
            # print(line_summary)

            # On détecte les cercles présents dans l'image
            circles = detect_circular_blobs(binary, show=True)

            # Détection des erreurs
            # =====================
            circuitOk = True

            # Les cerlces sont-ils pleins
            # print("\nDetected circles:")
            plainCircles = 0
            for circle in circles:
                plain, ratio = is_plain_circle(binary, circle)
                # print(circle, "plain:", plain, "fill_ratio:", round(ratio, 3))
                if plain == False:
                    circuitOk = False
                else:
                    plainCircles += 1
            
            # A t on bien les 3 cercles
            if len(circles) != 3:
                circuitOk = False

            # Les lignes sont elles pleines
            # print("\nDetected lines:")
            plainLines = 0
            vertical_lengths = []
            horizontal_lengths = []
            for line in lines:
                plain, info = is_plain_line(binary, line)
                # print(
                #     line["orientation"],
                #     line["points"],
                #     "angle:",
                #     round(line["angle"], 1),
                #     "plain:",
                #     plain,
                # )
                if plain == False:
                    circuitOk = False
                    break
                else:
                    plainLines += 1
                
                if line["orientation"] == "vertical":
                    vertical_lengths.append(info["length"])
                elif line["orientation"] == "horizontal":
                    horizontal_lengths.append(info["length"])

            # A t on bien 3 lignes verticales et 2 lignes horizontales
            nb_vertical = sum(
                1 for line in lines
                if line["orientation"] == "vertical"
            )

            nb_horizontal = sum(
                1 for line in lines
                if line["orientation"] == "horizontal"
            )

            if nb_vertical != 3 or nb_horizontal != 2:
                circuitOk = False

            # Les lignes sont elles de la même longueur
            if not same_lengths(vertical_lengths) or not same_lengths(horizontal_lengths):
                circuitOk = False 

            # Y a t il une ligne oblique
            oblique_lines = [
                line
                for line in lines
                if line["orientation"] == "oblique"
            ]
            if len(oblique_lines) > 0:
                circuitOk = False 

            rows.append({
                "class": class_name,
                "image_path": str(image_path),
                "image_name": image_path.name,
                "total_lines": len(lines),
                "complete_lines": plainLines,
                "total_circles": len(circles),
                "complete_circles": plainCircles,
                "circuit_complete": circuitOk
            })

            print(f"Circuit OK : {circuitOk}")

        except Exception as e:
            rows.append({
                "class": class_name,
                "image_path": str(image_path),
                "image_name": image_path.name,
                "total_lines": np.nan,
                "complete_lines": np.nan,
                "line_complete_rate": np.nan,
                "circuit_complete": False,
                "error": str(e)
            })
            print(f"Unable to test Circuit")

df = pd.DataFrame(rows)

# Dossier parent de chaque image
df["path"] = df["image_path"].apply(lambda p: str(Path(p).parent))

# Statut OK / KO
df["status"] = df["circuit_complete"].map({
    True: "OK",
    False: "KO"
})

summary = (
    df
    .groupby(["path", "status"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Sécurité si une colonne manque
if "OK" not in summary.columns:
    summary["OK"] = 0

if "KO" not in summary.columns:
    summary["KO"] = 0

summary["total"] = summary["OK"] + summary["KO"]
summary["ok_rate"] = summary["OK"] / summary["total"]
summary["ko_rate"] = summary["KO"] / summary["total"]

print(summary)