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
| **Données d'entraînement** | **1365(train**) | **1365 (train)** | **0** |
| **Temps train (CPU)** | **29.1s** | **530.6s** | **0** |
| **Latence inférence / image (CPU)** | **5972.8ms** | **56471.5ms** | ... |
| **Mémoire modèle (Mo)** | **2.2Mo** | **44.8Mo** | ~150 (CLIP) |
| **Accuracy attendue** | **0.276 (average train) / 0.495 (holdout)** | **0.401 (average train) / 0.489 (holdout)** | ... |
| **Coût € (training cloud)** | ~$0 (CPU local) | ~$0 (CPU local) | $0 |
| **Coût € (API)** | $0 (modèle local) | $0 | $0 (modèle local) |
| **Maintenance** | Réentraîner régulièrement | Réentraîner régulièrement | Aucune (prompts à raffiner) |

**Légende** :
- **Mesuré** : valeur obtenue dans notre implémentation
- _Estimé_ : valeur extrapolée de sources publiques (citer)

## Sources des estimations

> Pour les 2 options non implémentées, cite tes sources.

- Option ... : selon ... <source URL>
- Option ... : selon ... <source URL>

## Comparaison qualitative

| Aspect | Option A | Option B | Option C |
|---|---|---|---|
| **Quand préférer** | ... | ... | ... |
| **Quand éviter** | ... | ... | ... |
| **Domaine adapté** | ... | ... | ... |

---

*Comparatif produit en binôme — `<date>`.*
