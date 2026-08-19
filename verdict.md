# Verdict — Recommandation TechniMatic via Mistral
> 8 lignes maximum.
> Auteurs : `Julien D` × `Célia F` — Date : `19-08-2026`
**Recommandation** : Option B — transfer learning avec ResNet-18 pré-entraîné.
**Raison principale (chiffrée)** : le problème est très spécifique et les différences entre les défauts sont fines ; le pré-entraînement ImageNet de ResNet-18 fournit donc une base de représentations plus pertinente pour viser 90 % de précision. Dans le notebook, l'option B atteint 0,568 d'accuracy sur le holdout, avec 44,5 ms d'inférence par image et un modèle de 44,8 Mo ; elle est plus coûteuse que l'option A (0,606, 2,1 ms et 2,2 Mo), mais reste un compromis raisonnable pour gagner en capacité de généralisation.
**Condition de changement d'avis** : s'il y a peu de puissance de calcul et que la latence d'inférence doit être courte, l'option A serait plus adaptée ; CLIP pourrait devenir compétitif s'il acceptait directement une image.
---
*Verdict binôme — `Julien D` × `Célia F`, 19-08-2026.*
