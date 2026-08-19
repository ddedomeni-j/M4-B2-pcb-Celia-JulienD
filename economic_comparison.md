# Comparatif économique 3 approches — PCB Defect

> Document remis à **Inès Tabet** (Mistral) qui relaie à **TechniMatic**.
> Auteurs : `Julien D` × `Célia F` — Date : `31/08/26`

## Méthodologie

- **Option implémentée** : mesures **réelles** sur train + inférence
  - Julien a implémenté les options B et C
  - Célia a implémenté les options A et B
  - Comparaison des résultats sur l'option B
  - Pas d'estimation, analyse basée sur les mesures obtenues

- ⚠️ **Ordres de grandeur uniquement** : latence, temps d'entraînement et coût
  dépendent fortement du **hardware** (CPU/GPU, RAM, machine). On compare des
  **échelles relatives**, pas des vérités absolues.

## Tableau

| Critère | Option A (CNN scratch) | Option B (Transfer ResNet-18) | Option C (Zero-shot CLIP) |
|---|---|---|---|
| **Données d'entraînement requises** | ~1500 (train) | ~1500 (train) | **0** |
| **Données d'entraînement** | **1470(train**) | **1470 (train)** | **1470 (train)** |
| **Temps train (CPU)** | **308.9ms pour 16 epoch** | **5324ms pour 14 epoch** | **0** |
| **Latence inférence / image (CPU)** | **2.1ms** | **44.5ms** | ... |
| **Mémoire modèle (Mo)** | **2.2Mo** | **44.8Mo** (temps de save: 150 ms) | ~150 (CLIP) |
| **Accuracy attendue** | **0.606 (holdout)** | **0.568 (holdout)** | ... |
| **Coût € (training cloud)** | ~$0 (CPU local) | ~$0 (CPU local) | $0 |
| **Coût € (API)** | $0 (modèle local) | $0 | $0 (modèle local) |
| **Maintenance** | Réentraîner régulièrement | Réentraîner régulièrement | Aucune (prompts à raffiner) |

**Légende** :
- **Mesuré** : valeur obtenue dans notre implémentation
- _Estimé_ : valeur extrapolée de sources publiques (citer)

## Sources des estimations
N/A

## Comparaison qualitative

| Aspect | Option A (CNN) | Option B (transfer) | Option C (CLIP) |
|---|---|---|---|
| **Quand préférer** | Objectif pédagogique, tâches de segmentation (ex: photo aérienne), cas métier ultra-spécifique | Quand on veut spécialiser un problème sur un modèle précis avec un niveau d'exigence élevée | Quand les images sont proches des bases d'entrainement (situation ou objet de la vie courante) |
| **Quand éviter** | Si infrastructure ou performances limitées | Si l'option A est suffisante ou si problème de licence commerciale | Problème très spécifique et peu représentatif (c'est le cas ici) |
| **Domaine adapté** | Classification, segmentation | Classification, segmentation, détection d'objet | Démonstrateur rapide proche d'une situation ou d'un objet de la vie courante, détection d'objet courant |

---

*Comparatif produit en binôme — `19-08-2026`.*
