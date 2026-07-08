# Comparatif économique 3 approches — PCB Defect

> Document remis à **Inès Tabet** (Mistral) qui relaie à **TechniMatic**.
> Auteurs : `<prénom1>` × `<prénom2>` — Date : `<date>`

## Méthodologie

- **Option implémentée** : mesures **réelles** sur train + inférence
- **2 options non implémentées** : estimations argumentées via sources
  publiques (benchmarks PCB Defect, doc HuggingFace, articles cités)
- ⚠️ **Ordres de grandeur uniquement** : latence, temps d'entraînement et coût
  dépendent fortement du **hardware** (CPU/GPU, RAM, machine). On compare des
  **échelles relatives**, pas des vérités absolues.

## Tableau

| Critère | Option A (CNN scratch) | Option B (Transfer ResNet-18) | Option C (Zero-shot CLIP) |
|---|---|---|---|
| **Données d'entraînement requises** | ~1500 (train) | ~1500 (train) | **0** |
| **Temps train (CPU)** | ... | ... | **0** |
| **Latence inférence / image (CPU)** | ... | ... | ... |
| **Mémoire modèle (Mo)** | ... | ... | ~150 (CLIP) |
| **Accuracy attendue** | ... (mesuré OU estimé) | ... | ... |
| **Coût € (training cloud)** | ~$0 (CPU local) | ~$0 (CPU local) | $0 |
| **Coût € (API)** | $0 (modèle local) | $0 | $0 (modèle local) |
| **Maintenance** | Réentraîner régulièrement | Réentraîner régulièrement | Aucune (prompts à raffiner) |

**Légende** :
- **Mesuré** : valeur obtenue dans notre implémentation
- **Estimé** : valeur extrapolée de sources publiques (citer)

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
