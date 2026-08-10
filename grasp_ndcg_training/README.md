# GRASP SoftNDCG Training

This folder contains a separate experiment to train the best individual GRASP
architecture from the soft clinical assessment (`multihead_grasp`) with a
differentiable clinical NDCG surrogate.

The objective uses:

```text
predicted_score = max(0, 1 - p(resistance) - generation_penalty - AWaRe_penalty)
ideal_relevance = 1 - clinical_penalty for truly susceptible antibiotics, else 0
loss = 1 - SoftNDCG@1/@3/@5 + small BCE stabilizer
```

Default local/server pickle resolution:

```text
/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl
/Users/elenagarciaarroyo/Documents/Máster/Generative AI /project_2/COMBINED_MARISMA_DRIAMS_samples.pkl
```

Run:

```bash
python train_multihead_grasp_soft_ndcg.py \
  --output-dir results/multihead_grasp_soft_ndcg \
  --device auto
```

For a quick smoke test:

```bash
python train_multihead_grasp_soft_ndcg.py \
  --n-folds 2 \
  --max-epochs 1 \
  --patience 1 \
  --output-dir results/smoke_soft_ndcg
```

