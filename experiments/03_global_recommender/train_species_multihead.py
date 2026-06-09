# ============================================================
# NO-ENRICHMENT RECOMMENDERS FOR MALDI-TOF AMR PREDICTION
#
# Runs two controlled baseline experiments WITHOUT biological,
# epidemiological or clinical enrichment:
#
#   1) species_multihead_no_enrichment
#      - Shared MALDI backbone: 6000 -> 512 -> 128
#      - Species-specific MALDI head: 128 -> 64 -> 32
#      - Species-specific final recommender MLP
#      - Base recommender logit from species-adapted MALDI + antibiotic
#
#   2) implicit_species_hypernetwork_no_enrichment
#      - Global MALDI encoder: 6000 -> 512 -> 128 -> 64 -> 32
#      - Species embedding: species_id -> 16
#      - Species hypernetwork: 16 -> 64 -> 64 -> 32
#      - Trainable residual scale alpha
#      - z_final = z_maldi + alpha * delta_species
#      - Species-specific final recommender MLP
#      - Base recommender logit from species-corrected MALDI + antibiotic
#
# Core idea:
#
#   Multi-head:
#       z_final = SpeciesHeadMALDIEncoder(x, species_id)
#
#   Hypernetwork:
#       z_maldi = GlobalMALDIEncoder(x)
#       delta_species = HyperNetwork(species_emb)
#       z_final = z_maldi + alpha * delta_species
#
#   base_logit = SpeciesSpecificBaseMLP(
#       [z_final, drug_emb],
#       species_id
#   )
#
#   NOTE: the species embedding is NOT concatenated into the final recommender.
#         In the hypernetwork model it is used only internally to generate delta_species.
#
# Metrics saved:
#   - global micro AUC
#   - macro AUC by antibiotic
#   - micro AUC by species
#   - macro AUC by species
#       = AUC per antibiotic within species, then averaged
#   - crossed AUC species x antibiotic
#
# This script intentionally removes:
#   - ATC metadata
#   - mechanism metadata
#   - beta-lactamase metadata
#   - genus / gram metadata
#   - hospital / year metadata
#   - train prevalence priors
#   - enrichment embeddings
#   - enrichment logit correction
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

OUTPUT_FOLD_CSV = "no_enrichment_fold_results.csv"
OUTPUT_SPECIES_CSV = "no_enrichment_per_species.csv"
OUTPUT_ANTIBIOTIC_CSV = "no_enrichment_per_antibiotic.csv"
OUTPUT_SPECIES_ANTIBIOTIC_CSV = "no_enrichment_per_species_antibiotic.csv"

OUTPUT_SUMMARY_CSV = "no_enrichment_summary.csv"
OUTPUT_SPECIES_SUMMARY_CSV = "no_enrichment_per_species_summary.csv"
OUTPUT_ANTIBIOTIC_SUMMARY_CSV = "no_enrichment_per_antibiotic_summary.csv"
OUTPUT_SPECIES_ANTIBIOTIC_SUMMARY_CSV = "no_enrichment_per_species_antibiotic_summary.csv"

OUTPUT_MAPPING_JSON = "no_enrichment_mappings.json"

N_SPLITS = 5
RANDOM_STATE = 42

# Antibiotic filtering per fold
MIN_TRAIN_OBS_PER_ANTIBIOTIC = 50
MIN_VAL_OBS_PER_ANTIBIOTIC = 5

# Training
MAX_EPOCHS = 300
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 0.0

# GPU settings
USE_CUDA = torch.cuda.is_available()
ACCELERATOR = "gpu" if USE_CUDA else "cpu"
DEVICES = 1
PRECISION = "16-mixed" if USE_CUDA else "32-true"

if USE_CUDA:
    BATCH_SIZE = 1024
    VAL_BATCH_SIZE = 2048
    PRED_BATCH_SIZE = 512
    NUM_WORKERS = 4
else:
    BATCH_SIZE = 256
    VAL_BATCH_SIZE = 512
    PRED_BATCH_SIZE = 128
    NUM_WORKERS = 0

CPU_THREADS = min(8, os.cpu_count() or 1)
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(1)

# Architecture dimensions
#
# Multi-head MALDI path:
#   input_dim -> 512 -> 128 -> species-specific head 128 -> 64 -> 32
#
# Hypernetwork MALDI path:
#   input_dim -> 512 -> 128 -> 64 -> 32
#   species_id -> 16 -> hypernetwork 16 -> 64 -> 64 -> 32
#   z_final = z_maldi + alpha * delta_species
#
# Final recommender in BOTH models:
#   [z_final ; drug_emb] = 32 + 16 = 48
#   one species-specific MLP per species: 48 -> 32 -> 1
MALDI_EMB_DIM = 32
DRUG_EMB_DIM = 16
HYPERNET_SPECIES_EMB_DIM = 16
HYPERNET_HIDDEN_DIMS = [64, 64]
ALPHA_INIT = 0.05

RECOMMENDER_HIDDEN_DIMS = [32]

MODEL_MODES = [
    "species_multihead_no_enrichment",
    "implicit_species_hypernetwork_no_enrichment",
]

pl.seed_everything(RANDOM_STATE, workers=True)

if USE_CUDA:
    print("CUDA available:", torch.cuda.get_device_name(0), flush=True)
    torch.set_float32_matmul_precision("medium")
else:
    print("WARNING: CUDA is not available. Running on CPU.", flush=True)

print("Accelerator:", ACCELERATOR, flush=True)
print("Precision:", PRECISION, flush=True)
print("Batch size:", BATCH_SIZE, flush=True)
print("Val batch size:", VAL_BATCH_SIZE, flush=True)
print("Prediction batch size:", PRED_BATCH_SIZE, flush=True)
print("Num workers:", NUM_WORKERS, flush=True)


# ============================================================
# DATASET
# ============================================================

class NoEnrichmentRecDataset(Dataset):
    """
    Each item:
        (MALDI sample, species_id, antibiotic_id) -> AMR label

    There is NO enrichment information in this dataset.
    """

    def __init__(self, X, species_ids, amr):
        self.X = np.asarray(X)
        self.species_ids = np.asarray(species_ids, dtype=np.int64)
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
        species_id = int(self.species_ids[sample_idx])
        label = self.labels[idx]

        return (
            torch.tensor(self.X[sample_idx]).float(),
            torch.tensor(species_id).long(),
            torch.tensor(drug_id).long(),
            torch.tensor(label).float(),
        )


# ============================================================
# MODEL COMPONENTS
# ============================================================

class GlobalMALDIEncoder(nn.Module):
    """
    Global MALDI encoder shared by all species.

    Architecture:
        input_dim -> 512 -> 128 -> 64 -> 32

    Used by:
        implicit_species_hypernetwork_no_enrichment
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
            nn.GELU(),
        )

    def forward(self, maldi):
        return self.encoder(maldi.float())


class SpeciesHeadMALDIEncoder(nn.Module):
    """
    Shared MALDI backbone + one species-specific MALDI head per species.

    Architecture:
        Shared backbone:
            input_dim -> 512 -> 128

        Species-specific head:
            128 -> 64 -> 32

    Used by:
        species_multihead_no_enrichment
    """

    def __init__(self, input_dim, num_species, maldi_emb_dim=32):
        super().__init__()

        self.output_dim = maldi_emb_dim
        self.num_species = num_species

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.species_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(0.2),

                nn.Linear(64, maldi_emb_dim),
                nn.GELU(),
            )
            for _ in range(num_species)
        ])

    def forward(self, maldi, species_id):
        z_shared = self.backbone(maldi.float())
        species_id = species_id.long()

        # Important for mixed precision: under 16-mixed, Linear layers may
        # return float16 even if the input tensor is float32. Therefore, the
        # output buffer must be created with the dtype of the first head output,
        # not blindly with z_shared.dtype.
        out = None

        for sp in torch.unique(species_id):
            sp_int = int(sp.item())
            mask = species_id == sp
            head_out = self.species_heads[sp_int](z_shared[mask])

            if out is None:
                out = torch.empty(
                    size=(z_shared.shape[0], self.output_dim),
                    dtype=head_out.dtype,
                    device=head_out.device,
                )

            out[mask] = head_out

        return out


class SpeciesHyperNetwork(nn.Module):
    """
    Species-conditioned hypernetwork that generates a 32-dimensional
    residual correction for the global MALDI representation.

    Architecture:
        species_emb(16) -> 64 -> 64 -> 32
    """

    def __init__(self, species_emb_dim=16, hidden_dims=(64, 64), output_dim=32):
        super().__init__()

        sizes = [species_emb_dim] + list(hidden_dims) + [output_dim]
        layers = []

        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, species_emb):
        return self.network(species_emb)



class SpeciesSpecificRecommenderMLP(nn.Module):
    """
    One final recommender MLP per species.

    Each species-specific final MLP receives the SAME type of input:
        [z_final ; drug_emb] = 32 + 16 = 48

    Architecture of each species-specific MLP:
        48 -> 32 -> 1

    This makes the final classifier/recommender species-dependent WITHOUT
    concatenating the species embedding into the final input vector.
    """

    def __init__(self, num_species, input_dim=48, hidden_dims=(32,), output_dim=1):
        super().__init__()

        self.num_species = num_species
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.species_mlps = nn.ModuleList([
            self._make_mlp(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
            )
            for _ in range(num_species)
        ])

    def _make_mlp(self, input_dim, hidden_dims, output_dim):
        sizes = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []

        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        return nn.Sequential(*layers)

    def forward(self, x, species_id):
        species_id = species_id.long()

        # Important for mixed precision: under 16-mixed, each species-specific
        # MLP can return float16 while x may still be float32. Create the output
        # buffer from the first MLP output dtype to avoid dtype mismatch errors.
        out = None

        for sp in torch.unique(species_id):
            sp_int = int(sp.item())
            mask = species_id == sp
            mlp_out = self.species_mlps[sp_int](x[mask])

            if out is None:
                out = torch.empty(
                    size=(x.shape[0], self.output_dim),
                    dtype=mlp_out.dtype,
                    device=mlp_out.device,
                )

            out[mask] = mlp_out

        return out


class NoEnrichmentRecommender(pl.LightningModule):
    """
    Unified model for the two no-enrichment recommender architectures.

    1) species_multihead_no_enrichment:
        z_final = SpeciesHeadMALDIEncoder(MALDI, species_id)

    2) implicit_species_hypernetwork_no_enrichment:
        z_maldi = GlobalMALDIEncoder(MALDI)
        species_emb = Embedding(species_id)              # internal only
        delta_species = SpeciesHyperNetwork(species_emb)
        z_final = z_maldi + alpha * delta_species        # alpha is trainable

    In both cases the final recommender receives ONLY:
        final_logit = SpeciesSpecificMLP([z_final, drug_emb], species_id)

    Therefore the final input dimension is:
        32 + 16 = 48

    The final MLP itself is species-dependent:
        one independent 48 -> 32 -> 1 recommender head per species.

    The species embedding is NOT concatenated to the final recommender.
    Species information enters the final stage only by selecting the
    species-specific recommender MLP.
    """

    def __init__(
        self,
        num_feat,
        num_items,
        num_species,
        model_mode,
        maldi_emb_dim=32,
        drug_emb_dim=16,
        hypernet_species_emb_dim=16,
        hypernet_hidden_dims=(64, 64),
        alpha_init=0.05,
        hidden_dims=(32,),
        lr=1e-3,
        weight_decay=0.0,
    ):
        super().__init__()

        assert model_mode in [
            "species_multihead_no_enrichment",
            "implicit_species_hypernetwork_no_enrichment",
        ]

        self.save_hyperparameters()

        self.model_mode = model_mode
        self.lr = lr
        self.weight_decay = weight_decay
        self.maldi_emb_dim = maldi_emb_dim
        self.drug_emb_dim = drug_emb_dim

        if model_mode == "species_multihead_no_enrichment":
            self.maldi_encoder = SpeciesHeadMALDIEncoder(
                input_dim=num_feat,
                num_species=num_species,
                maldi_emb_dim=maldi_emb_dim,
            )
            self.species_embedding = None
            self.species_hypernetwork = None
            self.alpha = None
        else:
            self.maldi_encoder = GlobalMALDIEncoder(
                input_dim=num_feat,
                maldi_emb_dim=maldi_emb_dim,
            )
            self.species_embedding = nn.Embedding(
                num_embeddings=num_species,
                embedding_dim=hypernet_species_emb_dim,
            )
            self.species_hypernetwork = SpeciesHyperNetwork(
                species_emb_dim=hypernet_species_emb_dim,
                hidden_dims=hypernet_hidden_dims,
                output_dim=maldi_emb_dim,
            )
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self.drug_embedding = nn.Embedding(num_items, drug_emb_dim)

        # Final recommender input:
        #   z_final: 32
        #   drug_emb: 16
        #   total: 48
        self.input_dim = maldi_emb_dim + drug_emb_dim

        self.recommender_mlp = SpeciesSpecificRecommenderMLP(
            num_species=num_species,
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

    def _encode_maldi(self, maldi, species_id):
        if self.model_mode == "species_multihead_no_enrichment":
            return self.maldi_encoder(maldi, species_id)

        z_maldi = self.maldi_encoder(maldi)
        species_emb = self.species_embedding(species_id.long())
        delta_species = self.species_hypernetwork(species_emb)
        z_final = z_maldi + self.alpha * delta_species

        return z_final

    def get_alpha_value(self):
        if self.alpha is None:
            return np.nan
        return float(self.alpha.detach().cpu().item())

    def forward(self, maldi, species_id, drug_id):
        z_final = self._encode_maldi(maldi, species_id)
        drug_emb = self.drug_embedding(drug_id.long())

        x = torch.cat(
            [
                z_final,
                drug_emb,
            ],
            dim=-1,
        )

        logit = self.recommender_mlp(x, species_id)

        return logit

    def _shared_step(self, batch, stage):
        maldi, species_id, drug_id, labels = batch

        logits = self.forward(
            maldi=maldi,
            species_id=species_id,
            drug_id=drug_id,
        )

        labels = labels.view(-1, 1).float()

        loss = self.loss_fn(logits, labels)

        self.log(f"loss_{stage}", loss, prog_bar=True)
        self.log(f"logit_mean_{stage}", logits.detach().mean(), prog_bar=False)
        self.log(f"logit_abs_{stage}", logits.detach().abs().mean(), prog_bar=False)

        if self.alpha is not None:
            self.log(f"alpha_{stage}", self.alpha.detach(), prog_bar=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "tr")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


# ============================================================
# UTILS
# ============================================================

def clean_amr(amr):
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
        pin_memory=USE_CUDA,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )


def count_observed_with_two_classes(col):
    obs = col[~np.isnan(col)]
    n_obs = len(obs)
    has_two_classes = n_obs > 0 and len(np.unique(obs)) > 1
    return n_obs, has_two_classes


def select_valid_antibiotics(amr_train, amr_val):
    valid_cols = []

    for j in range(amr_train.shape[1]):
        n_train, train_two_classes = count_observed_with_two_classes(amr_train[:, j])
        n_val, val_two_classes = count_observed_with_two_classes(amr_val[:, j])

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
    y = np.asarray(y)
    p = np.asarray(p)

    if len(y) == 0:
        return np.nan

    if len(np.unique(y)) < 2:
        return np.nan

    return roc_auc_score(y, p)


def count_trainable_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ============================================================
# PREDICTION
# ============================================================

def predict_all_pairs(
    model,
    X,
    species_ids,
    num_items,
    batch_size=128,
):
    """
    Predict resistance probabilities for all validation samples and all selected antibiotics.

    Output:
        preds_matrix with shape (n_samples, num_items)
    """

    model.eval()
    device = model.device

    X = np.asarray(X)
    species_ids = np.asarray(species_ids, dtype=np.int64)

    n_samples = X.shape[0]
    preds_matrix = np.zeros((n_samples, num_items), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            bs = end - start

            maldi_batch = torch.tensor(
                X[start:end],
                dtype=torch.float32,
                device=device,
            )

            species_batch = torch.tensor(
                species_ids[start:end],
                dtype=torch.long,
                device=device,
            )

            drug_preds = []

            for drug_id in range(num_items):
                drug_batch = torch.full(
                    size=(bs,),
                    fill_value=drug_id,
                    dtype=torch.long,
                    device=device,
                )

                logits = model(
                    maldi=maldi_batch,
                    species_id=species_batch,
                    drug_id=drug_batch,
                )

                probs = torch.sigmoid(logits).view(-1)
                drug_preds.append(probs)

            drug_preds = torch.stack(drug_preds, dim=1)
            preds_matrix[start:end] = drug_preds.float().cpu().numpy()

    return preds_matrix


# ============================================================
# METRICS
# ============================================================

def compute_global_metrics(y_true, preds):
    """
    Computes:
        - global micro AUC over all observed sample-antibiotic pairs
        - macro antibiotic AUC:
            AUC per antibiotic, then mean across antibiotics
    """

    mask = ~np.isnan(y_true)

    if np.any(mask):
        global_micro_auc = safe_auc(y_true[mask], preds[mask])
    else:
        global_micro_auc = np.nan

    antibiotic_aucs = []

    for j in range(y_true.shape[1]):
        valid = ~np.isnan(y_true[:, j])
        auc_j = safe_auc(y_true[valid, j], preds[valid, j])

        if np.isfinite(auc_j):
            antibiotic_aucs.append(auc_j)

    macro_antibiotic_auc = (
        float(np.mean(antibiotic_aucs))
        if len(antibiotic_aucs) > 0
        else np.nan
    )

    return global_micro_auc, macro_antibiotic_auc, antibiotic_aucs


def compute_per_species_metrics(
    model_mode,
    fold,
    y_true,
    preds,
    species_ids,
    id_to_species,
):
    """
    Computes:
        - species_micro_auc:
            flatten all observed species-antibiotic pairs within species
        - species_macro_antibiotic_auc:
            AUC per antibiotic within species, then average
    """

    rows = []

    for sp_id in np.unique(species_ids):
        sp_mask = species_ids == sp_id

        y_sp = y_true[sp_mask]
        p_sp = preds[sp_mask]
        obs_mask = ~np.isnan(y_sp)

        n_samples = int(np.sum(sp_mask))
        n_pairs = int(np.sum(obs_mask))

        species_micro_auc = np.nan

        if n_pairs > 0:
            species_micro_auc = safe_auc(y_sp[obs_mask], p_sp[obs_mask])

        per_antibiotic_aucs = []
        per_antibiotic_n_pairs = []

        for j in range(y_sp.shape[1]):
            valid_j = ~np.isnan(y_sp[:, j])
            auc_j = safe_auc(y_sp[valid_j, j], p_sp[valid_j, j])

            if np.isfinite(auc_j):
                per_antibiotic_aucs.append(auc_j)
                per_antibiotic_n_pairs.append(int(np.sum(valid_j)))

        species_macro_antibiotic_auc = (
            float(np.mean(per_antibiotic_aucs))
            if len(per_antibiotic_aucs) > 0
            else np.nan
        )

        rows.append({
            "model_mode": model_mode,
            "fold": fold,
            "species_id": int(sp_id),
            "species": str(id_to_species[int(sp_id)]),
            "n_val_samples": n_samples,
            "n_val_pairs": n_pairs,
            "species_micro_auc": species_micro_auc,
            "species_macro_antibiotic_auc": species_macro_antibiotic_auc,
            "n_antibiotics_with_auc": len(per_antibiotic_aucs),
            "mean_pairs_per_antibiotic_with_auc": (
                float(np.mean(per_antibiotic_n_pairs))
                if len(per_antibiotic_n_pairs) > 0
                else np.nan
            ),
        })

    return rows


def compute_per_antibiotic_metrics(
    model_mode,
    fold,
    y_true,
    preds,
    selected_antibiotics,
):
    """
    Computes one global AUC per antibiotic.
    """

    rows = []

    for j, antibiotic_name in enumerate(selected_antibiotics):
        valid = ~np.isnan(y_true[:, j])

        n_pairs = int(np.sum(valid))
        auc = safe_auc(y_true[valid, j], preds[valid, j])

        rows.append({
            "model_mode": model_mode,
            "fold": fold,
            "antibiotic_id": int(j),
            "antibiotic": str(antibiotic_name),
            "n_val_pairs": n_pairs,
            "antibiotic_auc": auc,
        })

    return rows


def compute_per_species_antibiotic_metrics(
    model_mode,
    fold,
    y_true,
    preds,
    species_ids,
    id_to_species,
    selected_antibiotics,
):
    """
    Computes crossed AUC for each species-antibiotic pair.

    AUC is NaN when that pair does not have two classes in validation.
    """

    rows = []

    for sp_id in np.unique(species_ids):
        sp_mask = species_ids == sp_id
        species_name = str(id_to_species[int(sp_id)])

        y_sp = y_true[sp_mask]
        p_sp = preds[sp_mask]

        for j, antibiotic_name in enumerate(selected_antibiotics):
            y_pair = y_sp[:, j]
            p_pair = p_sp[:, j]

            valid = ~np.isnan(y_pair)
            n_pairs = int(np.sum(valid))

            auc = safe_auc(y_pair[valid], p_pair[valid])

            n_susceptible = np.nan
            n_resistant = np.nan

            if n_pairs > 0:
                n_susceptible = int(np.sum(y_pair[valid] == 0))
                n_resistant = int(np.sum(y_pair[valid] == 1))

            rows.append({
                "model_mode": model_mode,
                "fold": fold,
                "species_id": int(sp_id),
                "species": species_name,
                "antibiotic_id": int(j),
                "antibiotic": str(antibiotic_name),
                "n_val_pairs": n_pairs,
                "n_susceptible": n_susceptible,
                "n_resistant": n_resistant,
                "species_antibiotic_auc": auc,
            })

    return rows


def summarize_species_results(df_species):
    if len(df_species) == 0:
        return pd.DataFrame()

    rows = []

    for (model_mode, species), sub in df_species.groupby(["model_mode", "species"]):
        rows.append({
            "model_mode": model_mode,
            "species": species,
            "mean_species_micro_auc": sub["species_micro_auc"].mean(),
            "std_species_micro_auc": sub["species_micro_auc"].std(),
            "mean_species_macro_antibiotic_auc": sub["species_macro_antibiotic_auc"].mean(),
            "std_species_macro_antibiotic_auc": sub["species_macro_antibiotic_auc"].std(),
            "mean_n_val_pairs": sub["n_val_pairs"].mean(),
            "mean_n_antibiotics_with_auc": sub["n_antibiotics_with_auc"].mean(),
            "n_folds": sub["fold"].nunique(),
        })

    return pd.DataFrame(rows)


def summarize_antibiotic_results(df_antibiotics):
    if len(df_antibiotics) == 0:
        return pd.DataFrame()

    rows = []

    for (model_mode, antibiotic), sub in df_antibiotics.groupby(["model_mode", "antibiotic"]):
        rows.append({
            "model_mode": model_mode,
            "antibiotic": antibiotic,
            "mean_antibiotic_auc": sub["antibiotic_auc"].mean(),
            "std_antibiotic_auc": sub["antibiotic_auc"].std(),
            "mean_n_val_pairs": sub["n_val_pairs"].mean(),
            "n_folds": sub["fold"].nunique(),
        })

    return pd.DataFrame(rows)


def summarize_species_antibiotic_results(df_species_antibiotic):
    if len(df_species_antibiotic) == 0:
        return pd.DataFrame()

    rows = []

    for (model_mode, species, antibiotic), sub in df_species_antibiotic.groupby(
        ["model_mode", "species", "antibiotic"]
    ):
        rows.append({
            "model_mode": model_mode,
            "species": species,
            "antibiotic": antibiotic,
            "mean_species_antibiotic_auc": sub["species_antibiotic_auc"].mean(),
            "std_species_antibiotic_auc": sub["species_antibiotic_auc"].std(),
            "mean_n_val_pairs": sub["n_val_pairs"].mean(),
            "mean_n_susceptible": sub["n_susceptible"].mean(),
            "mean_n_resistant": sub["n_resistant"].mean(),
            "n_folds": sub["fold"].nunique(),
            "n_folds_with_valid_auc": sub["species_antibiotic_auc"].notna().sum(),
        })

    return pd.DataFrame(rows)


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

species_ids_all = np.asarray(
    [species_to_id[sp] for sp in species_labels],
    dtype=np.int64,
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


# ============================================================
# K-FOLD TRAINING
# ============================================================

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

fold_results = []
species_results = []
antibiotic_results = []
species_antibiotic_results = []

sample_indices = np.arange(n_samples)

global_mapping_payload = {
    "species_to_id": {str(k): int(v) for k, v in species_to_id.items()},
    "id_to_species": {str(k): str(v) for k, v in id_to_species.items()},
    "antibiotics_all": [str(a) for a in antibiotics_all],
    "model_modes": MODEL_MODES,
    "config": {
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "maldi_emb_dim": MALDI_EMB_DIM,
        "drug_emb_dim": DRUG_EMB_DIM,
        "hypernet_species_emb_dim": HYPERNET_SPECIES_EMB_DIM,
        "hypernet_hidden_dims": HYPERNET_HIDDEN_DIMS,
        "alpha_init": ALPHA_INIT,
        "recommender_hidden_dims": RECOMMENDER_HIDDEN_DIMS,
        "final_recommender_input_dim": MALDI_EMB_DIM + DRUG_EMB_DIM,
        "species_specific_final_recommender": True,
        "final_recommender_architecture_per_species": "48 -> 32 -> 1",
        "batch_size": BATCH_SIZE,
        "val_batch_size": VAL_BATCH_SIZE,
        "pred_batch_size": PRED_BATCH_SIZE,
        "precision": PRECISION,
        "accelerator": ACCELERATOR,
        "min_train_obs_per_antibiotic": MIN_TRAIN_OBS_PER_ANTIBIOTIC,
        "min_val_obs_per_antibiotic": MIN_VAL_OBS_PER_ANTIBIOTIC,
    },
    "fold_mappings": {},
}

for model_mode in MODEL_MODES:
    print("\n##################################################", flush=True)
    print(f"RUNNING MODEL MODE: {model_mode}", flush=True)
    print("##################################################", flush=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(sample_indices)):
        print("\n==================================================", flush=True)
        print(f"MODE: {model_mode} | FOLD {fold + 1}/{N_SPLITS}", flush=True)
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
            amr_val=amr_val_source,
        )

        if len(valid_cols) == 0:
            print("Skipping fold: no valid antibiotics", flush=True)
            continue

        selected_antibiotics = antibiotics_all[valid_cols]
        num_items = len(selected_antibiotics)

        print("Valid antibiotics:", num_items, flush=True)
        print("Antibiotics:", list(selected_antibiotics), flush=True)

        X_train = np.asarray(X_train_source)
        X_val = np.asarray(X_val_source)

        species_train = np.asarray(species_train_source)
        species_val = np.asarray(species_val_source)

        amr_train = amr_train_source[:, valid_cols]
        amr_val = amr_val_source[:, valid_cols]

        train_dataset = NoEnrichmentRecDataset(
            X=X_train,
            species_ids=species_train,
            amr=amr_train,
        )

        val_dataset = NoEnrichmentRecDataset(
            X=X_val,
            species_ids=species_val,
            amr=amr_val,
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
            shuffle=True,
        )

        loader_val = make_loader(
            dataset=val_dataset,
            batch_size=VAL_BATCH_SIZE,
            shuffle=False,
        )

        model = NoEnrichmentRecommender(
            num_feat=num_feat,
            num_items=num_items,
            num_species=num_species,
            model_mode=model_mode,
            maldi_emb_dim=MALDI_EMB_DIM,
            drug_emb_dim=DRUG_EMB_DIM,
            hypernet_species_emb_dim=HYPERNET_SPECIES_EMB_DIM,
            hypernet_hidden_dims=HYPERNET_HIDDEN_DIMS,
            alpha_init=ALPHA_INIT,
            hidden_dims=RECOMMENDER_HIDDEN_DIMS,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        print("Model input dim:", model.input_dim, flush=True)
        print("Final recommender: species-specific MLPs, one per species", flush=True)
        print("Final recommender per-species architecture: 48 -> 32 -> 1", flush=True)
        print("Trainable parameters:", count_trainable_parameters(model), flush=True)

        if model_mode == "implicit_species_hypernetwork_no_enrichment":
            print("Initial trainable alpha:", model.get_alpha_value(), flush=True)

        global_mapping_payload["fold_mappings"][f"{model_mode}_fold_{fold}"] = {
            "selected_antibiotics": [str(a) for a in selected_antibiotics],
            "valid_cols_original": [int(c) for c in valid_cols],
            "num_items": int(num_items),
        }

        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator=ACCELERATOR,
            devices=DEVICES,
            precision=PRECISION,
            callbacks=[
                EarlyStopping(
                    monitor="loss_val",
                    patience=PATIENCE,
                    mode="min",
                )
            ],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            deterministic=False,
        )

        trainer.fit(model, loader_train, loader_val)

        print("Predicting all validation sample-antibiotic pairs...", flush=True)

        preds_val = predict_all_pairs(
            model=model,
            X=X_val,
            species_ids=species_val,
            num_items=num_items,
            batch_size=PRED_BATCH_SIZE,
        )

        global_micro_auc, macro_antibiotic_auc, antibiotic_aucs = compute_global_metrics(
            y_true=amr_val,
            preds=preds_val,
        )

        species_rows = compute_per_species_metrics(
            model_mode=model_mode,
            fold=fold,
            y_true=amr_val,
            preds=preds_val,
            species_ids=species_val,
            id_to_species=id_to_species,
        )
        species_results.extend(species_rows)

        antibiotic_rows = compute_per_antibiotic_metrics(
            model_mode=model_mode,
            fold=fold,
            y_true=amr_val,
            preds=preds_val,
            selected_antibiotics=selected_antibiotics,
        )
        antibiotic_results.extend(antibiotic_rows)

        species_antibiotic_rows = compute_per_species_antibiotic_metrics(
            model_mode=model_mode,
            fold=fold,
            y_true=amr_val,
            preds=preds_val,
            species_ids=species_val,
            id_to_species=id_to_species,
            selected_antibiotics=selected_antibiotics,
        )
        species_antibiotic_results.extend(species_antibiotic_rows)

        species_micro_values = [
            r["species_micro_auc"]
            for r in species_rows
            if np.isfinite(r["species_micro_auc"])
        ]

        species_macro_values = [
            r["species_macro_antibiotic_auc"]
            for r in species_rows
            if np.isfinite(r["species_macro_antibiotic_auc"])
        ]

        mean_species_micro_auc = (
            float(np.mean(species_micro_values))
            if len(species_micro_values) > 0
            else np.nan
        )

        mean_species_macro_antibiotic_auc = (
            float(np.mean(species_macro_values))
            if len(species_macro_values) > 0
            else np.nan
        )

        print("Fold global/micro AUC:", global_micro_auc, flush=True)
        print("Fold macro antibiotic AUC:", macro_antibiotic_auc, flush=True)
        print("Fold mean species micro AUC:", mean_species_micro_auc, flush=True)
        print("Fold mean species macro-antibiotic AUC:", mean_species_macro_antibiotic_auc, flush=True)

        final_alpha = model.get_alpha_value()
        print("Fold final alpha:", final_alpha, flush=True)

        fold_results.append({
            "model_mode": model_mode,
            "fold": fold,
            "n_train_samples": X_train.shape[0],
            "n_val_samples": X_val.shape[0],
            "n_antibiotics": num_items,
            "n_train_pairs": len(train_dataset),
            "n_val_pairs": len(val_dataset),
            "model_input_dim": model.input_dim,
            "species_specific_final_recommender": True,
            "final_recommender_architecture_per_species": "48 -> 32 -> 1",
            "trainable_parameters": count_trainable_parameters(model),
            "maldi_emb_dim": MALDI_EMB_DIM,
            "hypernet_species_emb_dim": HYPERNET_SPECIES_EMB_DIM,
            "drug_emb_dim": DRUG_EMB_DIM,
            "final_alpha": final_alpha,
            "global_micro_auc": global_micro_auc,
            "macro_antibiotic_auc": macro_antibiotic_auc,
            "mean_species_micro_auc": mean_species_micro_auc,
            "mean_species_macro_antibiotic_auc": mean_species_macro_antibiotic_auc,
            "antibiotics": ";".join(map(str, selected_antibiotics)),
        })

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
df_species_antibiotic = pd.DataFrame(species_antibiotic_results)

print("\n==================================================")
print("FINAL FOLD RESULTS")
print("==================================================")
print(df_folds)

if len(df_folds) > 0:
    summary_rows = []

    for model_mode in MODEL_MODES:
        df_mode = df_folds[df_folds["model_mode"] == model_mode]

        summary_rows.append({
            "model_mode": model_mode,
            "mean_global_micro_auc": df_mode["global_micro_auc"].mean(),
            "std_global_micro_auc": df_mode["global_micro_auc"].std(),
            "mean_macro_antibiotic_auc": df_mode["macro_antibiotic_auc"].mean(),
            "std_macro_antibiotic_auc": df_mode["macro_antibiotic_auc"].std(),
            "mean_species_micro_auc": df_mode["mean_species_micro_auc"].mean(),
            "std_species_micro_auc": df_mode["mean_species_micro_auc"].std(),
            "mean_species_macro_antibiotic_auc": df_mode["mean_species_macro_antibiotic_auc"].mean(),
            "std_species_macro_antibiotic_auc": df_mode["mean_species_macro_antibiotic_auc"].std(),
            "mean_final_alpha": df_mode["final_alpha"].mean(),
            "std_final_alpha": df_mode["final_alpha"].std(),
            "mean_trainable_parameters": df_mode["trainable_parameters"].mean(),
            "n_folds": len(df_mode),
        })

    df_summary = pd.DataFrame(summary_rows)
else:
    df_summary = pd.DataFrame()

print("\n==================================================")
print("SUMMARY")
print("==================================================")
print(df_summary)

df_species_summary = summarize_species_results(df_species)
df_antibiotic_summary = summarize_antibiotic_results(df_antibiotics)
df_species_antibiotic_summary = summarize_species_antibiotic_results(df_species_antibiotic)

df_folds.to_csv(OUTPUT_FOLD_CSV, index=False)
df_species.to_csv(OUTPUT_SPECIES_CSV, index=False)
df_antibiotics.to_csv(OUTPUT_ANTIBIOTIC_CSV, index=False)
df_species_antibiotic.to_csv(OUTPUT_SPECIES_ANTIBIOTIC_CSV, index=False)

df_summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
df_species_summary.to_csv(OUTPUT_SPECIES_SUMMARY_CSV, index=False)
df_antibiotic_summary.to_csv(OUTPUT_ANTIBIOTIC_SUMMARY_CSV, index=False)
df_species_antibiotic_summary.to_csv(
    OUTPUT_SPECIES_ANTIBIOTIC_SUMMARY_CSV,
    index=False,
)

with open(OUTPUT_MAPPING_JSON, "w") as f:
    json.dump(global_mapping_payload, f, indent=4)

print("\nSaved:")
print(OUTPUT_FOLD_CSV)
print(OUTPUT_SPECIES_CSV)
print(OUTPUT_ANTIBIOTIC_CSV)
print(OUTPUT_SPECIES_ANTIBIOTIC_CSV)
print(OUTPUT_SUMMARY_CSV)
print(OUTPUT_SPECIES_SUMMARY_CSV)
print(OUTPUT_ANTIBIOTIC_SUMMARY_CSV)
print(OUTPUT_SPECIES_ANTIBIOTIC_SUMMARY_CSV)
print(OUTPUT_MAPPING_JSON)
