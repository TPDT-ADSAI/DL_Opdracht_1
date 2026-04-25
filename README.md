# DL_Opdracht_1 — Neural Estate

Prijsvoorspelling huizen met 4 modellen: Dense NN (tabulair), CNN (afbeelding), Transfer Learning (EfficientNetB0), Multimodaal (late fusion).

## Team

**Groep 1**

| Naam | Kaggle username |
|---|---|
| Parsa Fakhr | ctrlz123 |
| Thomas Kuijvenhoven | thomaskuijvenhoven |
| Ruben Derksen | rubenderksen |
| Teun De Maesschalck | teundemaesschalck |

**Kaggle leaderboard MAPE: 0.31579**

## Bestanden

- `main.ipynb` — volledige notebook
- `main.html` — gerenderde export voor inlevering
- `submissions/` — Kaggle submissions per model + ensemble
- `submission_ensemble_fc_heavy.csv` — beste submission (FC-heavy ensemble)

## Resultaten (OOF MAPE, 7-fold CV)

| Model | OOF MAPE | Ensemble-gewicht |
|---|---|---|
| FC (Dense, tabulair) | **0.2552** | 0.9089 |
| CNN (custom) | 6.6799 | 0.0000 |
| Transfer Learning (EfficientNetB0) | 1.6978 | 0.0062 |
| Multimodaal (late fusion) | 0.6270 | 0.0849 |
| **Ensemble (FC-heavy)** | **0.2559** | - |

## Wijzigingen n.a.v. feedback opdracht 1 (bijlage voor nakijker)

### 1. Dense NN (feedback: "Keuze van lossfunctie kon beter zijn. MAPE is de evaluatiemetric")

- Optuna-objective: MAPE-op-price (`mean_absolute_percentage_error` op `np.expm1(pred)`), hyperparameters geoptimaliseerd op exact de Kaggle-metric.
- Training-loss: MAE op log1p(price). Directe MAPE-loss (`mape_price_loss` via log-diff identity) numeriek instabiel. MAE op log is voor kleine relatieve errors wiskundig equivalent aan MAPE op price (`d log x / dx = 1/x`).
- Nieuw cel 21: bias-correctie (factor 1.097) tegen log-normale onderschatting.
- Nieuw cel 25: residual-plot + top-10 worst predictions per prijssegment.
- Nieuw cel 26: Huber vs MAE ablation (delta -0.01 binnen single-split-variantie).
- Resultaat: FC OOF MAPE 0.2739 → 0.2552.

### 2. CNN (feedback: "De data augmentatie hier is niet geschikt voor deze data")

- MixUp en Cutout/Random Erasing verwijderd. Reden: MixUp mengt 2 huizen + prijzen onrealistisch voor regressie, Cutout op 4-grid collage geeft geen ablation-winst en verhoogt variantie.
- Augmentatie teruggebracht tot geometrisch (h-flip, kleine shift, kleine zoom).
- Toegevoegd: seed-averaging (3 seeds/fold) + Test-Time Augmentation (h-flip).
- Nieuw cel 37: saliency-visualisatie op 3 huizen (goedkoopste, mediaan, duurste).
- Cel 35: ablation-tabel met motivatie verwijdering MixUp/Cutout.
- Resultaat CNN OOF MAPE: 4.66 → 6.68 (regressie door fold-variantie op kleine val-sets met dure outliers, niet door augmentatie-keuze).

### 3. Bevindingen + advies (feedback: "Advies voor opdrachtgevers/cliënten?")

- Hoofdstuk 6 Bevindingen toegevoegd met overzicht resultaten + Kaggle score.
- Advies-sectie opgesplitst per stakeholder: makelaar, taxateur, hypotheekverstrekker, huizenkopers/verkopers, met concrete rekenvoorbeelden (uren + euro-besparing per stakeholder).
- Toegevoegd: failure-mode analyse, kritische methodologie-reflectie (3 keuzes), 5 next-steps, beperkingen, operationele randvoorwaarden.
- Nieuw cel 55: per-prijssegment MAPE tabel.
- Nieuw cel 59: vergelijkende trade-off tabel (MAPE x params x train-tijd x interpretability x productie-readiness).

**Niet aangepast (al 5/5)**: EDA, Transfer Learning, Multimodaal model.
