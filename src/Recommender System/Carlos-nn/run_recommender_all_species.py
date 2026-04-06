import os
import pickle
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from pytorch_lightning.callbacks import EarlyStopping

from lib.NCF import NCF
from lib.RecDataset import RecDataset


# =========================
# CONFIG
# =========================
DATA_PATH = '/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS.pkl'

N_SPLITS = 5
BATCH_SIZE = 32
MAX_EPOCHS = 300
PATIENCE = 15

EMB_DIM_S = 30
EMB_DIM_I = 15
EMB_DIM_M = 10
HIDDEN_S = [500]

DEVICE = 'cpu'


# =========================
# LOAD DATA
# =========================
print("Loading dataset...", flush=True)
with open(DATA_PATH, 'rb') as f:
    payload = pickle.load(f)

X_all = payload["data"]
y_species_all = payload["label"]
amr_all = payload["amr"]
antibiotics = payload["antibiotics"]

species_list = np.unique(y_species_all)

print("Species found:", len(species_list), flush=True)


# =========================
# BUILD DATAFRAME
# =========================
def build_db_for_species(species):

    mask = (y_species_all == species)

    X = X_all[mask]
    amr = amr_all[mask]

    rows = []

    for i in range(X.shape[0]):
        for j in range(amr.shape[1]):

            value = amr[i, j]

            if value is None:
                continue
            if isinstance(value, float) and np.isnan(value):
                continue
            if value == -1:
                continue

            rows.append({
                "sample_id": i,
                "item_id": antibiotics[j],
                "resistance": int(value),
                "maldi": X[i],
                "sample_type": 0
            })

    df = pd.DataFrame(rows)

    # =========================
    # FILTRADO (IGUAL QUE CODIGO 1)
    # =========================
    counts = df.groupby('item_id')['resistance'].count()
    valid = counts[counts > 50].index

    valid_final = []
    for ab in valid:
        vals = df[df['item_id'] == ab]['resistance']
        if len(np.unique(vals)) >= 2:
            valid_final.append(ab)

    df = df[df['item_id'].isin(valid_final)].reset_index(drop=True)

    # Re-encode
    enc_items = LabelEncoder()
    df['item_id'] = enc_items.fit_transform(df['item_id'])

    return df, enc_items


# =========================
# FOLDS
# =========================
def create_sample_folds(sample_ids, n_splits=5, seed=42):

    unique_samples = np.unique(sample_ids)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []

    for train_idx, test_idx in kf.split(unique_samples):
        train_samples = unique_samples[train_idx]
        test_samples = unique_samples[test_idx]
        folds.append((train_samples, test_samples))

    return folds


# =========================
# TRAIN ONE SPECIES
# =========================
def train_species(species):

    print(f"\n====================")
    print(f"Species: {species}")
    print("====================", flush=True)

    db_rec, enc_items = build_db_for_species(species)

    if len(db_rec) < 100:
        print("Too few samples, skipping...", flush=True)
        return None

    num_items = db_rec['item_id'].nunique()
    num_meta = 1
    num_samples = db_rec['sample_id'].nunique()

    print("Samples:", num_samples, "Items:", num_items, flush=True)

    folds = create_sample_folds(db_rec['sample_id'].values, N_SPLITS)

    fold_micro = []
    fold_macro = []

    for fold in range(N_SPLITS):

        print(f"Fold {fold}", flush=True)

        train_samples, test_samples = folds[fold]
        val_samples, _ = folds[(fold + 1) % N_SPLITS]

        X_tr = db_rec[db_rec['sample_id'].isin(train_samples)]
        X_val = db_rec[db_rec['sample_id'].isin(val_samples)]
        X_tst = db_rec[db_rec['sample_id'].isin(test_samples)]

        maldis_tr = np.stack(X_tr['maldi'].values)
        maldis_val = np.stack(X_val['maldi'].values)
        maldis_tst = np.stack(X_tst['maldi'].values)

        meta_tr = X_tr['sample_type'].values
        meta_val = X_val['sample_type'].values
        meta_tst = X_tst['sample_type'].values

        drugs_tr = X_tr['item_id'].values
        drugs_val = X_val['item_id'].values
        drugs_tst = X_tst['item_id'].values

        res_tr = X_tr['resistance'].values
        res_val = X_val['resistance'].values
        res_tst = X_tst['resistance'].values

        loader_tr = DataLoader(RecDataset(maldis_tr, meta_tr, drugs_tr, res_tr), batch_size=BATCH_SIZE)
        loader_val = DataLoader(RecDataset(maldis_val, meta_val, drugs_val, res_val), batch_size=BATCH_SIZE)
        loader_tst = DataLoader(RecDataset(maldis_tst, meta_tst, drugs_tst, res_tst), batch_size=BATCH_SIZE)

        model = NCF(
            num_feat=maldis_tr.shape[1],
            num_items=num_items,
            num_meta=num_meta,
            sample_encoder='CNN',
            embedding_dim_samples=EMB_DIM_S,
            embedding_dim_items=EMB_DIM_I,
            embedding_dim_metadata=EMB_DIM_M,
            hidden_dim_samples=HIDDEN_S
        )

        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            callbacks=[EarlyStopping(monitor="loss_val", mode="min", patience=PATIENCE)],
            logger=False,
            enable_checkpointing=False
        )

        trainer.fit(model, loader_tr, loader_val)

        # ======================
        # PREDICCIONES
        # ======================
        model.eval()
        with torch.no_grad():
            preds = model(
                torch.tensor(maldis_tst).float(),
                torch.tensor(meta_tst).long(),
                torch.tensor(drugs_tst).long()
            ).numpy()

        # ======================
        # MICRO AUC
        # ======================
        if len(np.unique(res_tst)) > 1:
            auc_micro = roc_auc_score(res_tst, preds)
            fold_micro.append(auc_micro)

        # ======================
        # MACRO AUC
        # ======================
        antibiotic_aucs = []

        for d in np.unique(drugs_tst):
            mask = (drugs_tst == d)

            if len(np.unique(res_tst[mask])) > 1:
                auc_d = roc_auc_score(res_tst[mask], preds[mask])
                antibiotic_aucs.append(auc_d)

        if len(antibiotic_aucs) > 0:
            fold_macro.append(np.mean(antibiotic_aucs))

    return {
        "species": species,
        "auc_micro": np.mean(fold_micro),
        "auc_macro": np.mean(fold_macro),
        "auc_micro_std": np.std(fold_micro),
        "auc_macro_std": np.std(fold_macro)
    }


# =========================
# MAIN LOOP
# =========================
results = []

for sp in species_list:
    try:
        res = train_species(sp)
        if res:
            results.append(res)
    except Exception as e:
        print(f"Error in {sp}:", e)


# =========================
# SAVE RESULTS
# =========================
if len(results) == 0:
    print("No results generated!")
    exit()

df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="auc_macro", ascending=False)

print("\nFINAL RESULTS:")
print(df_results)

df_results.to_csv("results_code2_mlp.csv", index=False)