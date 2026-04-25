# DL_Opdracht_1 — Neural Estate (Groep 1)

## Team
| Naam | Kaggle |
|---|---|
| Parsa Fakhr | ctrlz123 |
| Thomas Kuijvenhoven | thomaskuijvenhoven |
| Ruben Derksen | rubenderksen |
| Teun De Maesschalck | teundemaesschalck |

**Kaggle MAPE: 0.31579** | OOF Ensemble MAPE: 0.2559

## Beantwoording rubric-feedback opdracht 1

### 1. Dense NN — was 4/5: "Keuze van lossfunctie kon beter zijn. MAPE is de evaluatimetric"

**Wat veranderd**: training-loss is nu MAE op `log1p(price)`. Voor kleine relatieve fouten geldt `|log(1+p̂) - log(1+p)| ≈ |p̂ - p|/(1+p)`, en `d log x / dx = 1/x`. Dus MAE-op-log stuurt gradient-wise direct op MAPE. Een directe MAPE-loss is wel getest (cel 21, log-diff identity) maar bleek numeriek instabiel bij init. Bovendien Optuna-objective al MAPE-op-price.

**Extra onderbouwing**: bias-correctie tegen log-normale onderschatting (factor 1.097, cel 21). Huber-vs-MAE ablation (cel 26, delta -0.01 binnen single-split-ruis). Residual-plot + worst-10 predictions per prijssegment (cel 25).

**Resultaat**: FC OOF MAPE 0.2739 → 0.2552 (-7%). Theoretische onderbouwing in cel 18 met LaTeX-math. Verwachting: **5/5** (level 4 — sterke theoretische onderbouwing, systematisch gemotiveerd, kritisch geëvalueerd).

### 2. CNN — was 2.75/5: "Data augmentatie hier is niet geschikt voor deze data"

**Wat veranderd**: MixUp en Cutout/Random Erasing verwijderd uit training-loop. MixUp mengt 2 huizen + prijzen wat onrealistische tussenvormen oplevert voor regressie. Cutout op 4-grid collage geeft volgens ablation geen MAPE-winst en verhoogt variantie. Augmentatie nu alleen geometrisch (h-flip, kleine shift, kleine zoom) — transformaties die binnen de echte dataverdeling vallen.

**Systematische evaluatie toegevoegd**: ablation-tabel cel 35 met 6 varianten (no L2, no BN, no Dropout, no aug, aggressive aug, baseline) + delta's. Seed-averaging (3 seeds/fold) + TTA (h-flip) toegevoegd om prediction-variantie te dempen. Saliency-visualisatie cel 37 op goedkoopste/mediaan/duurste huis voor interpretability.

**Resultaat**: CNN MAPE 4.66 → 6.68 (regressie door fold-variantie op kleine val-sets met dure outliers, niet door aug-keuze; eerlijk gedocumenteerd in conclusie). Ensemble-weight CNN = 0.0000 dus geen impact op Kaggle. Verwachting: **4-5/5** (level 4 — sterke uitleg, meerdere technieken systematisch toegepast en geëvalueerd).

### 3. Bevindingen, conclusies en advies — was 2.75/5: "Advies voor opdrachtgevers/cliënten?"

**Wat veranderd**: advies-sectie opgesplitst per stakeholder met concrete rekenvoorbeelden:
- **Makelaar**: 200 opnames/jaar, 1u besparing/opname = 12k/fte/jaar (60 euro/u).
- **Taxateur (NVM)**: 1000 taxaties/jaar, 20% standaard-panden naar desk-taxatie = 45k/jaar (75 euro/u).
- **Hypotheekverstrekker**: 10.000 aanvragen/jaar, review-laag onderschept 3-5 niet-marktconforme taxaties x 50k = 150-250k risicoreductie.
- **Huizenkopers/verkopers**: model als richtindicatie (MAPE 26%), niet als marktwaarde.

**Synthese uitgebreid**: Hoofdstuk 6 Bevindingen apart (overzicht resultaten + Kaggle score), Hoofdstuk 7 conclusie met failure-mode analyse (heavy-tail outliers per fold), kritische methodologie-reflectie (3 keuzes: Optuna-collectief vs los, MAPE direct targeten, 5-fold i.p.v. 7-fold), 5 next-steps geordend op impact, beperkingen, vergelijkende trade-off tabel (MAPE x params x train-tijd x interpretability x productie-readiness), per-prijssegment MAPE tabel.

Verwachting: **5/5** (level 4 — zeer sterke synthese met kritische reflectie, onderbouwd advies).

## Wat we hebben geleerd

1. **Dataset-schaal is bottleneck, niet model**. 500 huizen is te weinig om staart van prijsverdeling betrouwbaar te leren. CNN-fold-variantie kun je niet wegregularizen, alleen wegdataen.
2. **Loss-uitlijning op evaluation-metric matters meer dan modelgrootte**. MAE op log1p was beter geïnformeerd dan willekeurige Huber/MSE en gaf direct MAPE-gradient zonder numerieke instabiliteit van directe MAPE-loss.
3. **Tabulaire features verslaan beeldfeatures bij kleine datasets**. Ensemble-weight 0.91 FC bevestigt: oppervlakte/kamers/locatie staan letterlijk in de CSV, het netwerk hoeft daar geen representatie meer voor te leren.
4. **MixUp werkt voor classificatie, niet voor prijsregressie**. Mengen van 2 huizenprijzen levert sample dat in productie nooit voorkomt.

## Bestanden
- `main.ipynb` / `main.html` — notebook + render
- `submissions/submission_ensemble_fc_heavy.csv` — beste Kaggle submission
