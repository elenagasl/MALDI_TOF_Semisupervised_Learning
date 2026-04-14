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
DATA_PATH = '/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl'

N_SPLITS = 5
BATCH_SIZE = 32
MAX_EPOCHS = 300
PATIENCE = 15

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
hospital_all = payload["hospital"]
sample_type_all = payload["sample_type"]

# =========================
# CHECKS
# =========================
assert len(X_all) == len(y_species_all) == len(amr_all) == len(hospital_all) == len(sample_type_all)

print("Unique hospitals:", np.unique(hospital_all), flush=True)

species_list = np.unique(y_species_all)
print("Species found:", len(species_list), flush=True)


# =========================
# BUILD DATAFRAME
# =========================
def build_db_for_species(species):

    mask = (y_species_all == species)

    indices = np.where(mask)[0]   

    X = X_all[mask]
    amr = amr_all[mask]
    hospitals = hospital_all[mask]
    sample_types = sample_type_all[mask]

    rows = []

    for idx_local, idx_global in enumerate(indices):

        for j in range(amr.shape[1]):

            value = amr[idx_local, j]

            if value is None or (isinstance(value, float) and np.isnan(value)) or value == -1:
                continue

            stype = sample_types[idx_local]

            if stype is None or stype == "" or (isinstance(stype, float) and np.isnan(stype)):
                stype = "unknown"

            rows.append({
                "sample_id": idx_global,  
                "item_id": antibiotics[j],
                "resistance": int(value),
                "maldi": X[idx_local],
                "hospital": hospitals[idx_local],
                "sample_type": stype
            })

    df = pd.DataFrame(rows)

    # =========================
    # FILTRADO
    # =========================
    counts = df.groupby('item_id')['resistance'].count()
    valid = counts[counts > 50].index

    valid_final = []
    for ab in valid:
        vals = df[df['item_id'] == ab]['resistance']
        if len(np.unique(vals)) >= 2:
            valid_final.append(ab)

    df = df[df['item_id'].isin(valid_final)].reset_index(drop=True)

    # =========================
    # ENCODING
    # =========================
    enc_items = LabelEncoder()
    df['item_id'] = enc_items.fit_transform(df['item_id'])

    enc_sample = LabelEncoder()
    df["sample_type"] = enc_sample.fit_transform(df["sample_type"])

    return df, enc_items, enc_sample


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

    db_rec, enc_items, enc_sample = build_db_for_species(species)

    if len(db_rec) < 100:
        print("Too few samples, skipping...", flush=True)
        return None

    num_items = db_rec['item_id'].nunique()
    num_hospitals = int(np.max(hospital_all)) + 1   # 🔥 dinámico
    num_sample_types = db_rec['sample_type'].nunique()
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

        hosp_tr = X_tr['hospital'].values
        hosp_val = X_val['hospital'].values
        hosp_tst = X_tst['hospital'].values

        stype_tr = X_tr['sample_type'].values
        stype_val = X_val['sample_type'].values
        stype_tst = X_tst['sample_type'].values

        drugs_tr = X_tr['item_id'].values
        drugs_val = X_val['item_id'].values
        drugs_tst = X_tst['item_id'].values

        res_tr = X_tr['resistance'].values
        res_val = X_val['resistance'].values
        res_tst = X_tst['resistance'].values

        loader_tr = DataLoader(
            RecDataset(maldis_tr, hosp_tr, stype_tr, drugs_tr, res_tr),
            batch_size=BATCH_SIZE
        )

        loader_val = DataLoader(
            RecDataset(maldis_val, hosp_val, stype_val, drugs_val, res_val),
            batch_size=BATCH_SIZE
        )

        model = NCF(
            num_feat=maldis_tr.shape[1],
            num_items=num_items,
            num_hospitals=num_hospitals,
            num_sample_types=num_sample_types,
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
                torch.tensor(hosp_tst).long(),
                torch.tensor(stype_tst).long(),
                torch.tensor(drugs_tst).long()
            ).cpu().numpy()

        # ======================
        # MICRO AUC
        # ======================
        if len(np.unique(res_tst)) > 1:
            fold_micro.append(roc_auc_score(res_tst, preds))

        # ======================
        # MACRO AUC
        # ======================
        antibiotic_aucs = []

        for d in np.unique(drugs_tst):
            mask = (drugs_tst == d)

            if len(np.unique(res_tst[mask])) > 1:
                antibiotic_aucs.append(roc_auc_score(res_tst[mask], preds[mask]))

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

df_results.to_csv("results_code_mlp_metadata.csv", index=False)