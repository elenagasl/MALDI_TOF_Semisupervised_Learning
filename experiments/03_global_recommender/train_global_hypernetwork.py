# ============================================================
# GLOBAL RECOMMENDER WITH IMPLICIT SPECIES HYPERNETWORK
#
# Architecture:
#   1) Global MALDI encoder:
#          MALDI -> z_maldi
#
#   2) Species hypernetwork:
#          species_embedding -> delta_species
#
#   3) Species-adapted MALDI representation:
#          z_final = z_maldi + alpha * delta_species
#
#   4) Final recommender:
#          concat(z_final, species_embedding, antibiotic_embedding)
#          -> resistance logit
#
# Output files:
#   - implicit_species_fold_results.csv
#   - implicit_species_per_species.csv
#   - implicit_species_per_antibiotic.csv
#   - implicit_species_summary.csv
#   - implicit_species_mappings.json
# ============================================================

import os
import gc
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from pytorch_lightning.callbacks import EarlyStopping


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl"

OUTPUT_FOLD_CSV = "implicit_species_fold_results.csv"
OUTPUT_SPECIES_CSV = "implicit_species_per_species.csv"
OUTPUT_ANTIBIOTIC_CSV = "implicit_species_per_antibiotic.csv"
OUTPUT_SUMMARY_CSV = "implicit_species_summary.csv"
OUTPUT_MAPPING_JSON = "implicit_species_mappings.json"

N_SPLITS = 5
RANDOM_STATE = 42

# Antibiotic filtering per fold
MIN_TRAIN_OBS_PER_ANTIBIOTIC = 50
MIN_VAL_OBS_PER_ANTIBIOTIC = 5

# Training
BATCH_SIZE = 256
VAL_BATCH_SIZE = 512
MAX_EPOCHS = 300
PATIENCE = 15
LR = 1e-3

# Model dimensions
MALDI_EMB_DIM = 32
DRUG_EMB_DIM = 16
SPECIES_EMB_DIM = 16
HYPERNET_HIDDEN_DIM = 64
HIDDEN_DIMS = [128, 64]

# Dataloader
NUM_WORKERS = 0

# CPU settings
CPU_THREADS = min(8, os.cpu_count())
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(1)

# Reproducibility
pl.seed_everything(RANDOM_STATE, workers=True)

if torch.cuda.is_available():
    print("CUDA available:", torch.cuda.get_device_name(0), flush=True)
else:
    print("WARNING: CUDA is not available. Running on CPU.", flush=True)


# ============================================================
# DATASET
# ============================================================

class GlobalRecDataset(Dataset):
    """
    Each item:
        (sample, species, antibiotic) -> resistance

    Inputs:
        - MALDI spectrum
        - species_id
        - antibiotic_id

    Output:
        - AMR label: 0 susceptible, 1 resistant
    """

    def __init__(self, X, species_ids, amr):
        self.X = np.asarray(X)
        self.species_ids = np.asarray(species_ids)
        self.amr = np.asarray(amr)

        valid = (~np.isnan(self.amr)) & ((self.amr == 0) | (self.amr == 1))

        sample_idx, drug_idx = np.where(valid)

        labels = self.amr[sample_idx, drug_idx].astype(np.float32)

        self.sample_idx = sample_idx.astype(np.int64)
        self.drug_idx = drug_idx.astype(np.int64)
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample_idx = self.sample_idx[idx]
        drug_id = self.drug_idx[idx]
        label = self.labels[idx]

        maldi = self.X[sample_idx]
        species_id = self.species_ids[sample_idx]

        return (
            torch.tensor(maldi).float(),
            torch.tensor(species_id).long(),
            torch.tensor(drug_id).long(),
            torch.tensor(label).float()
        )


# ============================================================
# MODEL
# ============================================================

class GlobalMALDIEncoder(nn.Module):
    """
    Standard global MALDI encoder.

    This is intentionally NOT species-specific.

    Input:
        MALDI spectrum [batch_size, num_feat]

    Output:
        z_maldi [batch_size, maldi_emb_dim]
    """

    def __init__(self, input_dim, maldi_emb_dim=32):
        super().__init__()

        self.output_dim = maldi_emb_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(64, maldi_emb_dim),
            nn.GELU()
        )

    def forward(self, maldi):
        return self.encoder(maldi.float())


class SpeciesHypernetwork(nn.Module):
    """
    Species hypernetwork.

    It receives the species embedding and produces a species-specific
    correction vector with the SAME dimensionality as the MALDI embedding.

    Input:
        species_emb [batch_size, species_emb_dim]

    Output:
        delta_species [batch_size, maldi_emb_dim]
    """

    def __init__(
        self,
        species_emb_dim=16,
        maldi_emb_dim=32,
        hidden_dim=64
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(species_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, maldi_emb_dim)
        )

    def forward(self, species_emb):
        return self.net(species_emb)


class GlobalImplicitSpeciesNCF(pl.LightningModule):
    """
    Global recommender with implicit species-conditioned MALDI correction.

    Steps:
        1) MALDI -> global MALDI embedding
        2) species_id -> species embedding
        3) species embedding -> hypernetwork -> delta_species
        4) final MALDI embedding = global MALDI embedding + alpha * delta_species
        5) concat(final MALDI embedding, species embedding, antibiotic embedding)
        6) final MLP -> resistance logit
    """

    def __init__(
        self,
        num_feat,
        num_items,
        num_species,
        maldi_emb_dim=32,
        drug_emb_dim=16,
        species_emb_dim=16,
        hypernet_hidden_dim=64,
        hidden_dims=[128, 64],
        lr=1e-3
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.maldi_emb_dim = maldi_emb_dim
        self.drug_emb_dim = drug_emb_dim
        self.species_emb_dim = species_emb_dim

        # Global MALDI encoder
        self.maldi_encoder = GlobalMALDIEncoder(
            input_dim=num_feat,
            maldi_emb_dim=maldi_emb_dim
        )

        # Standard embeddings
        self.species_embedding = nn.Embedding(
            num_species,
            species_emb_dim
        )

        self.drug_embedding = nn.Embedding(
            num_items,
            drug_emb_dim
        )

        # Species hypernetwork:
        # species_embedding -> MALDI correction
        self.species_hypernetwork = SpeciesHypernetwork(
            species_emb_dim=species_emb_dim,
            maldi_emb_dim=maldi_emb_dim,
            hidden_dim=hypernet_hidden_dim
        )

        # Learnable scaling factor for the residual correction.
        # It starts small so the model begins close to the global recommender.
        self.alpha = nn.Parameter(torch.tensor(0.1))

        fusion_input_dim = (
            maldi_emb_dim
            + species_emb_dim
            + drug_emb_dim
        )

        sizes = [fusion_input_dim] + list(hidden_dims) + [1]

        layers = []

        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        self.mlp = nn.Sequential(*layers)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, maldi, species_id, drug_id):
        """
        maldi:
            [batch_size, num_feat]

        species_id:
            [batch_size]

        drug_id:
            [batch_size]

        returns:
            logits [batch_size, 1]
        """

        # 1) Global MALDI embedding
        z_maldi = self.maldi_encoder(maldi)

        # 2) Species embedding
        species_emb = self.species_embedding(species_id.long())

        # 3) Hypernetwork generates species correction
        delta_species = self.species_hypernetwork(species_emb)

        # 4) Residual species adaptation
        z_final = z_maldi + self.alpha * delta_species

        # 5) Antibiotic embedding
        drug_emb = self.drug_embedding(drug_id.long())

        # 6) Final recommender input
        x = torch.cat(
            [
                z_final,
                species_emb,
                drug_emb
            ],
            dim=-1
        )

        logits = self.mlp(x)

        return logits

    def training_step(self, batch, batch_idx):
        maldi, species_id, drug_id, labels = batch

        logits = self.forward(
            maldi=maldi,
            species_id=species_id,
            drug_id=drug_id
        )

        labels = labels.view(-1, 1).float()

        loss = self.loss_fn(logits, labels)

        self.log("loss_tr", loss, prog_bar=True)
        self.log("alpha", self.alpha.detach(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        maldi, species_id, drug_id, labels = batch

        logits = self.forward(
            maldi=maldi,
            species_id=species_id,
            drug_id=drug_id
        )

        labels = labels.view(-1, 1).float()

        loss = self.loss_fn(logits, labels)

        self.log("loss_val", loss, prog_bar=True)
        self.log("alpha_val", self.alpha.detach(), prog_bar=False)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ============================================================
# UTILS
# ============================================================

def clean_amr(amr):
    """
    Keeps only:
        0 = susceptible
        1 = resistant
        NaN = missing

    Everything else becomes NaN.
    """

    amr = amr.copy()

    valid = (amr == 0) | (amr == 1) | np.isnan(amr)

    amr[~valid] = np.nan

    return amr


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=False
    )


def count_observed_with_two_classes(col):
    obs = col[~np.isnan(col)]

    n_obs = len(obs)

    has_two_classes = False

    if n_obs > 0 and len(np.unique(obs)) > 1:
        has_two_classes = True

    return n_obs, has_two_classes


def select_valid_antibiotics(amr_train, amr_val):
    """
    Select antibiotics valid in a fold.

    Criteria:
        - enough observed labels in train
        - both classes present in train
        - enough observed labels in validation
        - both classes present in validation
    """

    valid_cols = []

    n_antibiotics = amr_train.shape[1]

    for j in range(n_antibiotics):

        train_col = amr_train[:, j]
        val_col = amr_val[:, j]

        n_train, train_two_classes = count_observed_with_two_classes(train_col)
        n_val, val_two_classes = count_observed_with_two_classes(val_col)

        if n_train < MIN_TRAIN_OBS_PER_ANTIBIOTIC:
            continue

        if not train_two_classes:
            continue

        if n_val < MIN_VAL_OBS_PER_ANTIBIOTIC:
            continue

        if not val_two_classes:
            continue

        valid_cols.append(j)

    return valid_cols


def safe_auc(y, p):
    if len(y) == 0:
        return np.nan

    if len(np.unique(y)) < 2:
        return np.nan

    return roc_auc_score(y, p)


def predict_all_pairs(model, X, species_ids, num_items, batch_size=128):
    """
    Predict resistance probabilities for all:

        samples x antibiotics

    Output:
        preds_matrix with shape (n_samples, num_items)
    """

    model.eval()

    device = model.device

    X = np.asarray(X)
    species_ids = np.asarray(species_ids)

    n_samples = X.shape[0]

    preds_matrix = np.zeros((n_samples, num_items), dtype=np.float32)

    with torch.no_grad():

        for start in range(0, n_samples, batch_size):

            end = min(start + batch_size, n_samples)

            maldi_batch = torch.tensor(
                X[start:end],
                dtype=torch.float32,
                device=device
            )

            species_batch = torch.tensor(
                species_ids[start:end],
                dtype=torch.long,
                device=device
            )

            batch_size_real = maldi_batch.shape[0]

            drug_preds = []

            for drug_id in range(num_items):

                drug_batch = torch.full(
                    size=(batch_size_real,),
                    fill_value=drug_id,
                    dtype=torch.long,
                    device=device
                )

                logits = model(
                    maldi=maldi_batch,
                    species_id=species_batch,
                    drug_id=drug_batch
                )

                probs = torch.sigmoid(logits).view(-1)

                drug_preds.append(probs)

            drug_preds = torch.stack(drug_preds, dim=1)

            preds_matrix[start:end] = drug_preds.cpu().numpy()

    return preds_matrix


def compute_global_metrics(y_true, preds):
    """
    Computes:
        - micro AUC over all observed sample-antibiotic pairs
        - macro AUC averaged across antibiotics
    """

    mask = ~np.isnan(y_true)

    auc_micro = np.nan
    auc_macro_antibiotic = np.nan

    if np.any(mask):

        y_flat = y_true[mask]
        p_flat = preds[mask]

        auc_micro = safe_auc(y_flat, p_flat)

    antibiotic_aucs = []

    for j in range(y_true.shape[1]):

        col = y_true[:, j]
        valid = ~np.isnan(col)

        auc_j = safe_auc(col[valid], preds[valid, j])

        if np.isfinite(auc_j):
            antibiotic_aucs.append(auc_j)

    if len(antibiotic_aucs) > 0:
        auc_macro_antibiotic = float(np.mean(antibiotic_aucs))

    return auc_micro, auc_macro_antibiotic, antibiotic_aucs


def compute_per_species_metrics(
    model_mode,
    fold,
    y_true,
    preds,
    species_ids,
    id_to_species
):
    rows = []

    unique_species_ids = np.unique(species_ids)

    for sp_id in unique_species_ids:

        sp_mask = species_ids == sp_id

        y_sp = y_true[sp_mask]
        p_sp = preds[sp_mask]

        obs_mask = ~np.isnan(y_sp)

        n_samples = int(np.sum(sp_mask))
        n_pairs = int(np.sum(obs_mask))

        auc = np.nan

        if n_pairs > 0:

            y_flat = y_sp[obs_mask]
            p_flat = p_sp[obs_mask]

            auc = safe_auc(y_flat, p_flat)

        rows.append(
            {
                "model_mode": model_mode,
                "fold": fold,
                "species_id": int(sp_id),
                "species": id_to_species[int(sp_id)],
                "n_val_samples": n_samples,
                "n_val_pairs": n_pairs,
                "auc": auc
            }
        )

    return rows


def compute_per_antibiotic_metrics(
    model_mode,
    fold,
    y_true,
    preds,
    selected_antibiotics
):
    rows = []

    for j, antibiotic_name in enumerate(selected_antibiotics):

        col = y_true[:, j]
        valid = ~np.isnan(col)

        n_pairs = int(np.sum(valid))

        auc = safe_auc(col[valid], preds[valid, j])

        rows.append(
            {
                "model_mode": model_mode,
                "fold": fold,
                "antibiotic_id": j,
                "antibiotic": str(antibiotic_name),
                "n_val_pairs": n_pairs,
                "auc": auc
            }
        )

    return rows


# ============================================================
# LOAD DATA
# ============================================================

print("Loading combined pickle...", flush=True)

with open(DATA_PATH, "rb") as f:
    payload = pickle.load(f)

X_all = np.asarray(payload["data"])
species_labels = np.asarray(payload["label"])
amr_all = clean_amr(np.asarray(payload["amr"]))
antibiotics_all = np.asarray(payload["antibiotics"])

if "hospital" in payload:
    hospital_all = np.asarray(payload["hospital"])
else:
    hospital_all = np.asarray(["unknown"] * X_all.shape[0])

n_samples = X_all.shape[0]
num_feat = X_all.shape[1]
num_antibiotics_total = amr_all.shape[1]

unique_species = np.unique(species_labels)

species_to_id = {
    sp: i for i, sp in enumerate(unique_species)
}

id_to_species = {
    i: sp for sp, i in species_to_id.items()
}

species_ids_all = np.array(
    [species_to_id[sp] for sp in species_labels],
    dtype=np.int64
)

num_species = len(unique_species)

print("Total samples:", n_samples, flush=True)
print("Total features:", num_feat, flush=True)
print("Total antibiotics:", num_antibiotics_total, flush=True)
print("Total species:", num_species, flush=True)

print("\nSpecies distribution:", flush=True)
unique_sp_ids, sp_counts = np.unique(species_ids_all, return_counts=True)

for sp_id, c in zip(unique_sp_ids, sp_counts):
    print(f"{id_to_species[int(sp_id)]}: {c} samples", flush=True)

print("\nHospital/source distribution:", flush=True)
unique_hospitals, hospital_counts = np.unique(hospital_all, return_counts=True)

for h, c in zip(unique_hospitals, hospital_counts):
    print(f"Hospital/source {h}: {c} samples", flush=True)

mapping_payload = {
    "species_to_id": {str(k): int(v) for k, v in species_to_id.items()},
    "id_to_species": {str(k): str(v) for k, v in id_to_species.items()},
    "antibiotics": [str(a) for a in antibiotics_all],
    "model": "implicit_species_hypernetwork",
    "maldi_emb_dim": MALDI_EMB_DIM,
    "species_emb_dim": SPECIES_EMB_DIM,
    "drug_emb_dim": DRUG_EMB_DIM,
    "hypernet_hidden_dim": HYPERNET_HIDDEN_DIM,
}

with open(OUTPUT_MAPPING_JSON, "w") as f:
    json.dump(mapping_payload, f, indent=4)

print("\nSaved mappings to:", OUTPUT_MAPPING_JSON, flush=True)


# ============================================================
# 5-FOLD TRAINING
# ============================================================

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_results = []
species_results = []
antibiotic_results = []

sample_indices = np.arange(n_samples)

MODEL_MODE = "implicit_species_hypernetwork"

print("\n##################################################", flush=True)
print(f"RUNNING MODEL MODE: {MODEL_MODE}", flush=True)
print("##################################################", flush=True)

for fold, (train_idx, val_idx) in enumerate(kf.split(sample_indices)):

    print("\n==================================================", flush=True)
    print(f"MODE: {MODEL_MODE} | FOLD {fold + 1}/{N_SPLITS}", flush=True)
    print("==================================================", flush=True)

    X_train_source = X_all[train_idx]
    X_val_source = X_all[val_idx]

    species_train_source = species_ids_all[train_idx]
    species_val_source = species_ids_all[val_idx]

    amr_train_source = amr_all[train_idx]
    amr_val_source = amr_all[val_idx]

    print("Train samples:", X_train_source.shape[0], flush=True)
    print("Validation samples:", X_val_source.shape[0], flush=True)

    valid_cols = select_valid_antibiotics(
        amr_train=amr_train_source,
        amr_val=amr_val_source
    )

    if len(valid_cols) == 0:
        print("Skipping fold: no valid antibiotics", flush=True)
        continue

    selected_antibiotics = antibiotics_all[valid_cols]

    X_train = np.asarray(X_train_source)
    X_val = np.asarray(X_val_source)

    species_train = np.asarray(species_train_source)
    species_val = np.asarray(species_val_source)

    amr_train = amr_train_source[:, valid_cols]
    amr_val = amr_val_source[:, valid_cols]

    num_items = amr_train.shape[1]

    print("Valid antibiotics:", num_items, flush=True)
    print("Antibiotics:", list(selected_antibiotics), flush=True)

    train_dataset = GlobalRecDataset(
        X=X_train,
        species_ids=species_train,
        amr=amr_train
    )

    val_dataset = GlobalRecDataset(
        X=X_val,
        species_ids=species_val,
        amr=amr_val
    )

    if len(train_dataset) == 0:
        print("Skipping fold: empty train dataset", flush=True)
        continue

    if len(val_dataset) == 0:
        print("Skipping fold: empty val dataset", flush=True)
        continue

    print("Train pairs:", len(train_dataset), flush=True)
    print("Validation pairs:", len(val_dataset), flush=True)

    loader_train = make_loader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    loader_val = make_loader(
        dataset=val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False
    )

    model = GlobalImplicitSpeciesNCF(
        num_feat=num_feat,
        num_items=num_items,
        num_species=num_species,
        maldi_emb_dim=MALDI_EMB_DIM,
        drug_emb_dim=DRUG_EMB_DIM,
        species_emb_dim=SPECIES_EMB_DIM,
        hypernet_hidden_dim=HYPERNET_HIDDEN_DIM,
        hidden_dims=HIDDEN_DIMS,
        lr=LR
    )

    fusion_dim = MALDI_EMB_DIM + SPECIES_EMB_DIM + DRUG_EMB_DIM

    print("Fusion input dim:", fusion_dim, flush=True)
    print("Initial alpha:", float(model.alpha.detach().cpu()), flush=True)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",
        devices=1,
        callbacks=[
            EarlyStopping(
                monitor="loss_val",
                patience=PATIENCE,
                mode="min"
            )
        ],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True
    )

    trainer.fit(model, loader_train, loader_val)

    print("Final alpha:", float(model.alpha.detach().cpu()), flush=True)

    print("Predicting all validation sample-antibiotic pairs...", flush=True)

    preds_val = predict_all_pairs(
        model=model,
        X=X_val,
        species_ids=species_val,
        num_items=num_items,
        batch_size=128
    )

    auc_global, auc_macro_antibiotic, antibiotic_aucs = compute_global_metrics(
        y_true=amr_val,
        preds=preds_val
    )

    print("Fold global/micro AUC:", auc_global, flush=True)
    print("Fold macro antibiotic AUC:", auc_macro_antibiotic, flush=True)

    fold_results.append(
        {
            "model_mode": MODEL_MODE,
            "fold": fold,
            "n_train_samples": X_train.shape[0],
            "n_val_samples": X_val.shape[0],
            "n_antibiotics": num_items,
            "n_train_pairs": len(train_dataset),
            "n_val_pairs": len(val_dataset),
            "fusion_input_dim": fusion_dim,
            "maldi_emb_dim": MALDI_EMB_DIM,
            "species_emb_dim": SPECIES_EMB_DIM,
            "drug_emb_dim": DRUG_EMB_DIM,
            "hypernet_hidden_dim": HYPERNET_HIDDEN_DIM,
            "final_alpha": float(model.alpha.detach().cpu()),
            "global_auc": auc_global,
            "macro_antibiotic_auc": auc_macro_antibiotic,
            "antibiotics": ";".join(map(str, selected_antibiotics))
        }
    )

    species_rows = compute_per_species_metrics(
        model_mode=MODEL_MODE,
        fold=fold,
        y_true=amr_val,
        preds=preds_val,
        species_ids=species_val,
        id_to_species=id_to_species
    )

    species_results.extend(species_rows)

    antibiotic_rows = compute_per_antibiotic_metrics(
        model_mode=MODEL_MODE,
        fold=fold,
        y_true=amr_val,
        preds=preds_val,
        selected_antibiotics=selected_antibiotics
    )

    antibiotic_results.extend(antibiotic_rows)

    del model
    del trainer
    del loader_train
    del loader_val
    del train_dataset
    del val_dataset

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# SAVE RESULTS
# ============================================================

df_folds = pd.DataFrame(fold_results)
df_species = pd.DataFrame(species_results)
df_antibiotics = pd.DataFrame(antibiotic_results)

print("\n==================================================")
print("FINAL FOLD RESULTS")
print("==================================================")
print(df_folds)

if len(df_folds) > 0:

    df_summary = pd.DataFrame(
        [
            {
                "model_mode": MODEL_MODE,
                "mean_global_auc": df_folds["global_auc"].mean(),
                "std_global_auc": df_folds["global_auc"].std(),
                "mean_macro_antibiotic_auc": df_folds["macro_antibiotic_auc"].mean(),
                "std_macro_antibiotic_auc": df_folds["macro_antibiotic_auc"].std(),
                "mean_final_alpha": df_folds["final_alpha"].mean(),
                "std_final_alpha": df_folds["final_alpha"].std(),
                "n_folds": len(df_folds)
            }
        ]
    )

    print("\n==================================================")
    print("SUMMARY")
    print("==================================================")
    print(df_summary)

else:
    df_summary = pd.DataFrame()

df_folds.to_csv(OUTPUT_FOLD_CSV, index=False)
df_species.to_csv(OUTPUT_SPECIES_CSV, index=False)
df_antibiotics.to_csv(OUTPUT_ANTIBIOTIC_CSV, index=False)
df_summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

print("\nSaved:")
print(OUTPUT_FOLD_CSV)
print(OUTPUT_SPECIES_CSV)
print(OUTPUT_ANTIBIOTIC_CSV)
print(OUTPUT_SUMMARY_CSV)
print(OUTPUT_MAPPING_JSON)