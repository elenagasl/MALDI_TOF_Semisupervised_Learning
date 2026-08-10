# Clinical Assessment

Final clinical ranking analysis for the OOD experiments. This folder has been
reduced to the last analysis block: clinical NDCG, resistant/susceptible
recommendation percentages, and higher-generation antibiotic use.

## Kept Scripts

- `wide_panel_clinical_assessment.py`: computes wide-panel NDCG and rank-level
  safety metrics for raw and clinically reranked antibiotic recommendations.
- `recommendation_percentages.py`: computes interpretable percentages for
  recommended antibiotics, including true susceptible, true resistant, and
  higher-generation use.
- `compare_models_and_ensembles.py`: prepares model and ensemble threshold
  comparisons used by percentage summaries.
- `run_clinical_assessment.py`: shared constants and utilities for clinical
  scoring.
- `generations.py`: antibiotic generation metadata.
- `spectra.py`: antibiotic AWaRe/category metadata used by clinical penalties.

## Clinical Ranking

Predicted resistance probabilities are converted into recommendation scores.
Antibiotics predicted as resistant are excluded from the clinical ranking, and
susceptible candidates are penalized when they are broader-spectrum or belong to
less preferred stewardship categories.

```text
clinical_score = max(0, 1 - p_resistance - generation_penalty - AWaRe_penalty)
```

The ideal ranking gives positive relevance only to truly susceptible
antibiotics. NDCG is then reported at the selected ranks.

## Typical Run

From this folder:

```bash
python wide_panel_clinical_assessment.py \
  --results-root ../ood_evaluation/results \
  --output-dir results/wide_panel_clinical_assessment

python compare_models_and_ensembles.py \
  --results-root ../ood_evaluation/results \
  --output-dir results/model_comparison

python recommendation_percentages.py \
  --results-root ../ood_evaluation/results \
  --comparison-dir results/model_comparison \
  --output-dir results/recommendation_percentages
```
