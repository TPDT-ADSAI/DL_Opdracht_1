# DL_Opdracht_1 — Neural Estate

Prijsvoorspelling huizen met 4 modellen: Dense NN (tabulair), CNN (afbeelding), Transfer Learning (EfficientNetB0), Multimodaal (late fusion).

Hoofdbestanden:
- `main.ipynb` — volledige notebook
- `main.html` — gerenderde export voor inlevering
- `submissions/` — Kaggle submissions per model + ensemble

## Wijzigingen n.a.v. feedback opdracht 1 (bijlage voor nakijker)

Korte samenvatting van wat er t.o.v. de eerste inlevering is gewijzigd en waarom.

### 1. Dense NN (feedback: "Keuze van lossfunctie kon beter zijn. MAPE is de evaluatiemetric")

- Optuna-objective was en blijft MAPE-op-price (`mean_absolute_percentage_error` op `np.expm1(pred)`), dus hyperparameters worden geoptimaliseerd op exact de Kaggle-metric.
- Training-loss is MAE op log1p(price). Een expliciete MAPE-loss is getest (`mape_price_loss` via log-diff identity) maar bleek numeriek instabiel (gradient te klein bij initialisatie). MAE op log is voor kleine relatieve errors wiskundig equivalent aan MAPE op price (`d log x / dx = 1/x`), dus het stuurt impliciet op de evaluatiemetric en is robuuster in de training.
- Nieuw in cel 20: bias-correctie (`expm1` van gemiddelde log-price is kleiner dan arithmetische mean price bij log-normale verdeling). Factor `y_raw.mean() / oof.mean()` corrigeert de systematische onderschatting. In deze run: factor 1.097.
- Nieuw in cel 24: residual-plot + top-10 worst predictions als kritische evaluatie per prijssegment. Laat zien dat het model de staart van de verdeling onderschat.
- Nieuw in cel 25: Huber vs MAE ablation voor onderbouwing van de loss-keuze.
- Resultaat: FC OOF MAPE is nu 0.3105 (was 0.2739, hoger door bias-correctie die wel de systematische underfit corrigeert maar niet per definitie MAPE minimaliseert, een bewuste trade-off richting kritische evaluatie en productie-bruikbaarheid).

### 2. CNN (feedback: "De data augmentatie hier is niet geschikt voor deze data")

- MixUp en Cutout/Random Erasing zijn verwijderd. Reden: mengen van 2 huizen met hun prijzen (MixUp) levert voor prijsregressie onrealistische tussenvormen, en het wegmaskeren van collage-delen (Cutout) haalt bij een 4-grid kamer-collage prijs-informatie weg.
- Augmentatie is teruggebracht tot geometrische transformaties binnen de dataverdeling (h-flip, kleine shift, kleine zoom).
- Toegevoegd: Seed-averaging (2 seeds per fold) + Test-Time Augmentation (h-flip) om prediction-variantie te reduceren.
- Nieuw in cel 36: Grad-CAM / saliency visualisatie op 3 huizen (goedkoopste, mediaan, duurste) om de werking van het model inzichtelijk te maken.
- Cel 34 ablation-tabel bijgewerkt met motivatie voor verwijdering van MixUp/Cutout.
- Resultaat: CNN OOF MAPE is nu 3.8047 (was 4.663).

### 3. Bevindingen, conclusies, advies (feedback: "Advies voor opdrachtgevers/clienten?")

- Advies-sectie opgesplitst per stakeholder (makelaar, taxateur, hypotheekverstrekker) met concrete rekenvoorbeelden (jaarlijkse besparing in uren en euros per stakeholder).
- Toegevoegd: failure-mode analyse op per-fold uitschieters, kritische methodologie-reflectie (drie concrete keuzes die ik anders zou maken), en een toekomstig-werk sectie met 5 concrete next-steps.
- Nieuw in cel 53: per-prijssegment MAPE tabel die laat zien waar elk model goed of slecht presteert.
- Nieuw in cel 55: vergelijkende trade-off tabel (MAPE x params x train-tijd x interpretability x productie-readiness).

**Niet aangepast (al 5/5)**: EDA, Transfer Learning, Multimodaal model.
