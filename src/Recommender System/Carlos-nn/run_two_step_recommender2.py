import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl"

OUTPUT_DIR = "calibrated_two_stage_context_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SPLITS = 2
VAL_SIZE = 0.2

BATCH_SIZE = 64
PRED_BATCH_SIZE = 512

MAX_EPOCHS_STAGE1 = 300
MAX_EPOCHS_STAGE2 = 300

PATIENCE_STAGE1 = 15
PATIENCE_STAGE2 = 15

LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-3

DRUG_EMB_DIM = 32

# MLP más profunda para MALDI:
# si MALDI tiene ~6000 features:
# 6000 -> 2048 -> 1024 -> 512 -> 256 -> 128
MALDI_HIDDEN_DIMS = [2048, 1024, 512, 256, 128]
MALDI_EMB_DIM = 128

STAGE1_HIDDEN_DIMS = [256, 128, 64]
STAGE2_HIDDEN_DIMS = [256, 128, 64]

USE_CHECKPOINT = True

# Correlaciones en train
MIN_CORR_PAIR = 30

# Calibración
MIN_CALIBRATION_OBS = 30

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE, flush=True)

torch.backends.cudnn.benchmark = True


# ============================================================
# EXPERIMENTS
# ============================================================
EXPERIMENTS = [
    {
        "name": "calibrated_threshold_10_90",
        "context_mode": "threshold",
        "low_threshold": 0.10,
        "high_threshold": 0.90,
        "top_k": None,
        "corr_kind": None,
    },
    {
        "name": "calibrated_topk_3",
        "context_mode": "topk",
        "low_threshold": None,
        "high_threshold": None,
        "top_k": 3,
        "corr_kind": None,
    },
    {
        "name": "calibrated_topk_5",
        "context_mode": "topk",
        "low_threshold": None,
        "high_threshold": None,
        "top_k": 5,
        "corr_kind": None,
    },
    {
        "name": "calibrated_corr_pos_topk_3",
        "context_mode": "corr_topk",
        "low_threshold": None,
        "high_threshold": None,
        "top_k": 3,
        "corr_kind": "positive",
    },
    {
        "name": "calibrated_corr_pos_topk_5",
        "context_mode": "corr_topk",
        "low_threshold": None,
        "high_threshold": None,
        "top_k": 5,
        "corr_kind": "positive",
    },
    {
        "name": "calibrated_corr_abs_topk_3",
        "context_mode": "corr_topk",
        "low_threshold": None,
        "high_threshold": None,
        "top_k": 3,
        "corr_kind": "absolute",
    },
    {
        "name": "calibrated_corr_abs_topk_5",
        "context_mode": "corr_topk",
        "low_threshold": None,
        "high_threshold": None,
        "top_k": 5,
        "corr_kind": "absolute",
    },
]


# ============================================================
# LOAD DATA
# ============================================================
print("Loading dataset...", flush=True)

with open(DATA_PATH, "rb") as f:
    payload = pickle.load(f)

X_all = payload["data"]
y_species_all = payload["label"]
amr_all = payload["amr"]
antibiotics = payload["antibiotics"]

species_list = np.unique(y_species_all)


# ============================================================
# BASIC UTILITIES
# ============================================================
def safe_name(x):
    return str(x).replace("/", "_").replace(" ", "_")


def sigmoid_np(x):
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def build_species_data(species):

    species_mask = (y_species_all == species)

    X = X_all[species_mask]
    amr = amr_all[species_mask].copy()

    valid_values = (amr == 0) | (amr == 1) | np.isnan(amr)
    amr[~valid_values] = np.nan

    if len(X) < 100:
        return None

    valid_cols = []

    for j in range(amr.shape[1]):
        col = amr[:, j]
        col = col[~np.isnan(col)]

        if len(col) > 50 and len(np.unique(col)) > 1:
            valid_cols.append(j)

    if len(valid_cols) == 0:
        return None

    X = np.asarray(X)
    amr = amr[:, valid_cols]

    antibiotic_names = np.asarray(antibiotics)[valid_cols]

    return X, amr, antibiotic_names


def create_folds(n_samples, n_splits):

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    return list(kf.split(np.arange(n_samples)))


def split_train_val_indices(train_idx, val_size, seed):

    train_inner_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_size,
        random_state=seed,
        shuffle=True
    )

    return train_inner_idx, val_idx


# ============================================================
# DATASETS
# ============================================================
class AMRVectorDataset(Dataset):
    """
    Dataset para Stage 1.

    Devuelve:
        maldi
        labels: AMR con NaN -> 0
        mask: 1 donde hay valor real observado
    """

    def __init__(self, maldi, amr):
        self.maldi = np.asarray(maldi)
        self.amr = np.asarray(amr)

        if self.maldi.shape[0] != self.amr.shape[0]:
            raise ValueError("maldi and amr must have same number of samples.")

    def __len__(self):
        return self.maldi.shape[0]

    def __getitem__(self, idx):
        x = self.maldi[idx]
        y = self.amr[idx].copy()

        mask = ~np.isnan(y)

        labels = y.copy()
        labels[~mask] = 0

        return (
            torch.tensor(x).float(),
            torch.tensor(labels).float(),
            torch.tensor(mask).float()
        )


class Stage2ContextDataset(Dataset):
    """
    Dataset para Stage 2.

    Devuelve:
        maldi
        labels
        mask
        context_probs: pseudo-antibiograma calibrado de Stage 1
    """

    def __init__(self, maldi, amr, context_probs):
        self.maldi = np.asarray(maldi)
        self.amr = np.asarray(amr)
        self.context_probs = np.asarray(context_probs)

        if self.maldi.shape[0] != self.amr.shape[0]:
            raise ValueError("maldi and amr must have same number of samples.")

        if self.context_probs.shape != self.amr.shape:
            raise ValueError(
                f"context_probs shape {self.context_probs.shape} "
                f"must match amr shape {self.amr.shape}"
            )

    def __len__(self):
        return self.maldi.shape[0]

    def __getitem__(self, idx):
        x = self.maldi[idx]
        y = self.amr[idx].copy()
        probs = self.context_probs[idx]

        mask = ~np.isnan(y)

        labels = y.copy()
        labels[~mask] = 0

        return (
            torch.tensor(x).float(),
            torch.tensor(labels).float(),
            torch.tensor(mask).float(),
            torch.tensor(probs).float()
        )


def make_stage1_loader(X, amr, batch_size, shuffle, num_workers):

    dataset = AMRVectorDataset(
        maldi=X,
        amr=amr
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def make_stage2_loader(X, amr, context_probs, batch_size, shuffle, num_workers):

    dataset = Stage2ContextDataset(
        maldi=X,
        amr=amr,
        context_probs=context_probs
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


# ============================================================
# MODEL COMPONENTS
# ============================================================
class MaldiEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dims, emb_dim):
        super().__init__()

        dims = [input_dim] + list(hidden_dims)

        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.2))

        if dims[-1] != emb_dim:
            layers.append(nn.Linear(dims[-1], emb_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()

        dims = [input_dim] + list(hidden_dims) + [output_dim]

        layers = []

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# STAGE 1 MODEL
# ============================================================
class Stage1Recommender(pl.LightningModule):
    """
    Stage 1:

    MALDI embedding + antibiotic embedding
    -> logit de resistencia por antibiótico.

    Produce un pseudo-antibiograma completo.
    """

    def __init__(
        self,
        num_feat,
        num_items,
        drug_emb_dim=32,
        maldi_hidden_dims=None,
        maldi_emb_dim=128,
        hidden_dims=None,
        lr=1e-3
    ):
        super().__init__()

        self.save_hyperparameters()

        self.num_items = num_items
        self.lr = lr

        if maldi_hidden_dims is None:
            maldi_hidden_dims = [2048, 1024, 512, 256, 128]

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.maldi_encoder = MaldiEncoder(
            input_dim=num_feat,
            hidden_dims=maldi_hidden_dims,
            emb_dim=maldi_emb_dim
        )

        self.drug_embedding = nn.Embedding(
            num_items,
            drug_emb_dim
        )

        self.stage1_mlp = MLP(
            input_dim=maldi_emb_dim + drug_emb_dim,
            hidden_dims=hidden_dims,
            output_dim=1
        )

        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def masked_bce(self, logits, labels, mask):

        loss_matrix = self.loss_fn(logits, labels.float())

        loss = (loss_matrix * mask.float()).sum() / (
            mask.float().sum() + 1e-8
        )

        return loss

    def get_drug_emb_all(self, device):

        drug_ids = torch.arange(
            self.num_items,
            dtype=torch.long,
            device=device
        )

        return self.drug_embedding(drug_ids)

    def forward(self, maldi):

        batch_size = maldi.shape[0]
        device = maldi.device

        maldi_emb = self.maldi_encoder(maldi)

        drug_emb = self.get_drug_emb_all(device)

        maldi_expanded = maldi_emb.unsqueeze(1).expand(
            batch_size,
            self.num_items,
            maldi_emb.shape[-1]
        )

        drug_expanded = drug_emb.unsqueeze(0).expand(
            batch_size,
            self.num_items,
            drug_emb.shape[-1]
        )

        x = torch.cat(
            [maldi_expanded, drug_expanded],
            dim=-1
        )

        x = x.reshape(batch_size * self.num_items, -1)

        logits = self.stage1_mlp(x)

        logits = logits.view(batch_size, self.num_items)

        return logits

    def training_step(self, batch, batch_idx):

        maldi, labels, mask = [x.to(self.device) for x in batch]

        logits = self.forward(maldi)

        loss = self.masked_bce(
            logits=logits,
            labels=labels,
            mask=mask
        )

        self.log("loss_tr", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        maldi, labels, mask = [x.to(self.device) for x in batch]

        logits = self.forward(maldi)

        loss = self.masked_bce(
            logits=logits,
            labels=labels,
            mask=mask
        )

        self.log("loss_val", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ============================================================
# STAGE 2 MODEL
# ============================================================
class Stage2ContextPredictor(pl.LightningModule):
    """
    Stage 2:

    MALDI embedding + antibiotic embedding + pseudo-contexto calibrado
    -> logit final por antibiótico.

    Context modes:
        threshold:
            usa p <= low como 0 y p >= high como 1.

        topk:
            usa los k antibióticos más confiados globalmente.

        corr_topk:
            para cada target, usa los k antibióticos con mayor:
                confidence(pred) × correlation_with_target
    """

    def __init__(
        self,
        num_feat,
        num_items,
        context_mode,
        low_threshold=None,
        high_threshold=None,
        top_k=None,
        corr_kind=None,
        corr_matrix=None,
        drug_emb_dim=32,
        maldi_hidden_dims=None,
        maldi_emb_dim=128,
        hidden_dims=None,
        lr=1e-3
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["corr_matrix"])

        self.num_items = num_items
        self.context_mode = context_mode
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.top_k = top_k
        self.corr_kind = corr_kind
        self.lr = lr

        if maldi_hidden_dims is None:
            maldi_hidden_dims = [2048, 1024, 512, 256, 128]

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.maldi_encoder = MaldiEncoder(
            input_dim=num_feat,
            hidden_dims=maldi_hidden_dims,
            emb_dim=maldi_emb_dim
        )

        self.drug_embedding = nn.Embedding(
            num_items,
            drug_emb_dim
        )

        stage2_input_dim = (
            maldi_emb_dim +
            drug_emb_dim +
            num_items +   # context_values
            num_items     # context_mask
        )

        self.stage2_mlp = MLP(
            input_dim=stage2_input_dim,
            hidden_dims=hidden_dims,
            output_dim=1
        )

        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        if corr_matrix is None:
            corr_matrix = np.zeros((num_items, num_items), dtype=np.float32)

        corr_tensor = torch.tensor(corr_matrix).float()

        self.register_buffer("corr_matrix", corr_tensor)

    def masked_bce(self, logits, labels, mask):

        loss_matrix = self.loss_fn(logits, labels.float())

        loss = (loss_matrix * mask.float()).sum() / (
            mask.float().sum() + 1e-8
        )

        return loss

    def get_drug_emb_all(self, device):

        drug_ids = torch.arange(
            self.num_items,
            dtype=torch.long,
            device=device
        )

        return self.drug_embedding(drug_ids)

    def expand_context_per_target(self, values_base, mask_base):

        batch_size, num_items = values_base.shape
        device = values_base.device

        values = values_base.unsqueeze(1).expand(
            batch_size,
            num_items,
            num_items
        ).clone()

        masks = mask_base.unsqueeze(1).expand(
            batch_size,
            num_items,
            num_items
        ).clone()

        eye = torch.eye(num_items, device=device).unsqueeze(0)
        target_exclusion = 1.0 - eye

        values = values * target_exclusion
        masks = masks * target_exclusion

        return values, masks

    def build_threshold_context(self, context_probs):

        low = self.low_threshold
        high = self.high_threshold

        resistant = context_probs >= high
        susceptible = context_probs <= low

        confident = resistant | susceptible

        values_base = resistant.float()
        mask_base = confident.float()

        return self.expand_context_per_target(
            values_base=values_base,
            mask_base=mask_base
        )

    def build_topk_context(self, context_probs):

        batch_size, num_items = context_probs.shape

        k = self.top_k

        if k is None or k <= 0:
            values_base = torch.zeros_like(context_probs)
            mask_base = torch.zeros_like(context_probs)

            return self.expand_context_per_target(
                values_base=values_base,
                mask_base=mask_base
            )

        k = min(k, num_items)

        confidence = torch.abs(context_probs - 0.5) * 2.0

        _, idx = torch.topk(
            confidence,
            k=k,
            dim=1
        )

        mask_base = torch.zeros_like(context_probs)
        mask_base.scatter_(1, idx, 1.0)

        values_base = (context_probs >= 0.5).float()
        values_base = values_base * mask_base

        return self.expand_context_per_target(
            values_base=values_base,
            mask_base=mask_base
        )

    def build_corr_topk_context(self, context_probs):
        """
        Contexto target-specific basado en correlaciones de train.

        Para cada target j y cada posible contexto k:

            score(j, k) = confidence(k) * relevance(j, k)

        donde relevance puede ser:
            positive: max(phi, 0)
            absolute: abs(phi)

        Luego se seleccionan los top-k contextos para cada target.
        """

        batch_size, num_items = context_probs.shape
        device = context_probs.device

        k = self.top_k

        if k is None or k <= 0:
            values = torch.zeros(
                batch_size,
                num_items,
                num_items,
                device=device
            )

            masks = torch.zeros_like(values)

            return values, masks

        k = min(k, num_items - 1)

        confidence = torch.abs(context_probs - 0.5) * 2.0

        if self.corr_kind == "positive":
            relevance = torch.clamp(self.corr_matrix, min=0.0)

        elif self.corr_kind == "absolute":
            relevance = torch.abs(self.corr_matrix)

        else:
            raise ValueError(
                f"corr_kind must be 'positive' or 'absolute'. Got {self.corr_kind}"
            )

        relevance = relevance.clone()

        eye = torch.eye(num_items, device=device)
        relevance = relevance * (1.0 - eye)

        # score shape:
        # confidence: [B, N_context]
        # relevance:  [N_target, N_context]
        # score:      [B, N_target, N_context]
        score = confidence.unsqueeze(1) * relevance.unsqueeze(0)

        # Evitar seleccionar contextos con score 0
        positive_score_mask = score > 0

        _, idx = torch.topk(
            score,
            k=k,
            dim=2
        )

        masks = torch.zeros_like(score)
        masks.scatter_(2, idx, 1.0)

        masks = masks * positive_score_mask.float()

        values_base = (context_probs >= 0.5).float()

        values = values_base.unsqueeze(1).expand(
            batch_size,
            num_items,
            num_items
        ).clone()

        values = values * masks

        return values, masks

    def build_context_per_target(self, context_probs):

        if self.context_mode == "threshold":
            return self.build_threshold_context(context_probs)

        if self.context_mode == "topk":
            return self.build_topk_context(context_probs)

        if self.context_mode == "corr_topk":
            return self.build_corr_topk_context(context_probs)

        raise ValueError(f"Unknown context_mode: {self.context_mode}")

    def forward(self, maldi, context_probs):

        batch_size = maldi.shape[0]
        device = maldi.device

        maldi_emb = self.maldi_encoder(maldi)

        context_values, context_mask = self.build_context_per_target(
            context_probs=context_probs
        )

        drug_emb = self.get_drug_emb_all(device)

        maldi_expanded = maldi_emb.unsqueeze(1).expand(
            batch_size,
            self.num_items,
            maldi_emb.shape[-1]
        )

        drug_expanded = drug_emb.unsqueeze(0).expand(
            batch_size,
            self.num_items,
            drug_emb.shape[-1]
        )

        x = torch.cat(
            [
                maldi_expanded,
                drug_expanded,
                context_values,
                context_mask,
            ],
            dim=-1
        )

        x = x.reshape(batch_size * self.num_items, -1)

        logits = self.stage2_mlp(x)

        logits = logits.view(batch_size, self.num_items)

        return logits, context_mask

    def training_step(self, batch, batch_idx):

        maldi, labels, mask, context_probs = [
            x.to(self.device) for x in batch
        ]

        logits, _ = self.forward(
            maldi=maldi,
            context_probs=context_probs
        )

        loss = self.masked_bce(
            logits=logits,
            labels=labels,
            mask=mask
        )

        self.log("loss_tr", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        maldi, labels, mask, context_probs = [
            x.to(self.device) for x in batch
        ]

        logits, _ = self.forward(
            maldi=maldi,
            context_probs=context_probs
        )

        loss = self.masked_bce(
            logits=logits,
            labels=labels,
            mask=mask
        )

        self.log("loss_val", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ============================================================
# METRICS
# ============================================================
def compute_metrics(y_true, preds):

    valid_mask = ~np.isnan(y_true)

    auc_micro = np.nan
    auc_macro = np.nan

    if np.any(valid_mask) and len(np.unique(y_true[valid_mask])) > 1:
        auc_micro = roc_auc_score(
            y_true[valid_mask],
            preds[valid_mask]
        )

    antibiotic_aucs = []

    for j in range(y_true.shape[1]):

        col = y_true[:, j]
        valid = ~np.isnan(col)

        if np.sum(valid) > 0 and len(np.unique(col[valid])) > 1:
            antibiotic_aucs.append(
                roc_auc_score(
                    col[valid],
                    preds[valid, j]
                )
            )

    if len(antibiotic_aucs) > 0:
        auc_macro = float(np.mean(antibiotic_aucs))

    return auc_micro, auc_macro


def compute_context_coverage(y_true, context_mask_per_target):
    """
    context_mask_per_target tiene shape:
        [n_samples, num_targets, num_context_antibiotics]

    Calculamos qué fracción del vector contextual está visible
    para los targets realmente evaluables.
    """

    target_valid = ~np.isnan(y_true)

    if np.sum(target_valid) == 0:
        return np.nan

    num_items = y_true.shape[1]

    visible = context_mask_per_target[target_valid].sum()

    total_possible = np.sum(target_valid) * (num_items - 1)

    return float(visible / (total_possible + 1e-8))


# ============================================================
# CORRELATION MATRIX
# ============================================================
def compute_phi_correlation_matrix(amr_train, min_pair=30):

    num_items = amr_train.shape[1]

    W = np.zeros((num_items, num_items), dtype=np.float32)
    N = np.zeros((num_items, num_items), dtype=np.int32)

    for i in range(num_items):
        for j in range(i + 1, num_items):

            xi = amr_train[:, i]
            xj = amr_train[:, j]

            valid = ~np.isnan(xi) & ~np.isnan(xj)

            n_valid = int(np.sum(valid))

            N[i, j] = n_valid
            N[j, i] = n_valid

            if n_valid < min_pair:
                continue

            a = xi[valid].astype(int)
            b = xj[valid].astype(int)

            if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
                continue

            n11 = np.sum((a == 1) & (b == 1))
            n10 = np.sum((a == 1) & (b == 0))
            n01 = np.sum((a == 0) & (b == 1))
            n00 = np.sum((a == 0) & (b == 0))

            denom = np.sqrt(
                (n11 + n10) *
                (n01 + n00) *
                (n11 + n01) *
                (n10 + n00)
            )

            if denom == 0:
                phi = 0.0
            else:
                phi = (n11 * n00 - n10 * n01) / denom

            W[i, j] = phi
            W[j, i] = phi

    return W, N


# ============================================================
# CALIBRATION
# ============================================================
class PerAntibioticPlattCalibrator:
    """
    Calibración por antibiótico usando Platt scaling:

        p_calibrada = sigmoid(a_j * logit_j + b_j)

    Se ajusta con validation.
    """

    def __init__(self, min_obs=30):
        self.min_obs = min_obs
        self.a = None
        self.b = None
        self.was_fitted = None

    def fit(self, logits_val, y_val):

        num_items = y_val.shape[1]

        self.a = np.ones(num_items, dtype=np.float32)
        self.b = np.zeros(num_items, dtype=np.float32)
        self.was_fitted = np.zeros(num_items, dtype=bool)

        for j in range(num_items):

            y = y_val[:, j]
            valid = ~np.isnan(y)

            n_valid = int(np.sum(valid))

            if n_valid < self.min_obs:
                continue

            y_valid = y[valid].astype(int)

            if len(np.unique(y_valid)) < 2:
                continue

            x_valid = logits_val[valid, j].reshape(-1, 1)

            try:
                clf = LogisticRegression(
                    C=1e6,
                    solver="lbfgs",
                    max_iter=1000
                )

                clf.fit(x_valid, y_valid)

                self.a[j] = float(clf.coef_[0][0])
                self.b[j] = float(clf.intercept_[0])
                self.was_fitted[j] = True

            except Exception:
                self.a[j] = 1.0
                self.b[j] = 0.0
                self.was_fitted[j] = False

        return self

    def transform(self, logits):

        calibrated_logits = logits * self.a.reshape(1, -1) + self.b.reshape(1, -1)

        probs = sigmoid_np(calibrated_logits)

        return probs

    def to_dataframe(self, antibiotic_names=None):

        rows = []

        for j in range(len(self.a)):

            if antibiotic_names is None:
                ab = f"drug_{j}"
            else:
                ab = antibiotic_names[j]

            rows.append({
                "antibiotic_idx": j,
                "antibiotic": ab,
                "platt_a": self.a[j],
                "platt_b": self.b[j],
                "was_fitted": bool(self.was_fitted[j])
            })

        return pd.DataFrame(rows)


# ============================================================
# PREDICTION HELPERS
# ============================================================
def predict_stage1_logits(model, X, batch_size=PRED_BATCH_SIZE):

    model.eval()

    all_logits = []

    with torch.no_grad():

        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])

            x_batch = torch.tensor(
                X[start:end]
            ).float().to(model.device)

            logits = model(x_batch)

            all_logits.append(logits.cpu().numpy())

    return np.vstack(all_logits)


def predict_stage2(model, X, context_probs, batch_size=PRED_BATCH_SIZE):

    model.eval()

    all_probs = []
    all_context_masks = []

    with torch.no_grad():

        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])

            x_batch = torch.tensor(
                X[start:end]
            ).float().to(model.device)

            context_batch = torch.tensor(
                context_probs[start:end]
            ).float().to(model.device)

            logits, context_mask = model(
                maldi=x_batch,
                context_probs=context_batch
            )

            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_context_masks.append(context_mask.cpu().numpy())

    return np.vstack(all_probs), np.vstack(all_context_masks)


# ============================================================
# TRAIN STAGE 1
# ============================================================
def train_stage1(species, fold, X_tr, amr_tr, X_val, amr_val, num_items):

    print(
        f"\nTraining Stage 1 | species={species} | fold={fold}",
        flush=True
    )

    loader_tr = make_stage1_loader(
        X=X_tr,
        amr=amr_tr,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    loader_val = make_stage1_loader(
        X=X_val,
        amr=amr_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    model = Stage1Recommender(
        num_feat=X_tr.shape[1],
        num_items=num_items,
        drug_emb_dim=DRUG_EMB_DIM,
        maldi_hidden_dims=MALDI_HIDDEN_DIMS,
        maldi_emb_dim=MALDI_EMB_DIM,
        hidden_dims=STAGE1_HIDDEN_DIMS,
        lr=LR_STAGE1
    )

    callbacks = [
        EarlyStopping(
            monitor="loss_val",
            patience=PATIENCE_STAGE1,
            mode="min"
        )
    ]

    checkpoint_callback = None

    if USE_CHECKPOINT:
        checkpoint_callback = ModelCheckpoint(
            monitor="loss_val",
            mode="min",
            save_top_k=1,
            filename=(
                f"{safe_name(species)}_stage1_fold{fold}"
                + "-{epoch:02d}-{loss_val:.4f}"
            )
        )

        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS_STAGE1,
        accelerator="gpu" if DEVICE == "cuda" else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=USE_CHECKPOINT
    )

    trainer.fit(model, loader_tr, loader_val)

    if USE_CHECKPOINT and checkpoint_callback is not None:
        best_path = checkpoint_callback.best_model_path

        if best_path is not None and best_path != "":
            print(f"Loading best Stage 1 checkpoint: {best_path}", flush=True)
            model = Stage1Recommender.load_from_checkpoint(best_path)
            model = model.to(DEVICE)

    return model


# ============================================================
# TRAIN STAGE 2
# ============================================================
def train_stage2(
    experiment,
    species,
    fold,
    X_tr,
    amr_tr,
    context_probs_tr,
    X_val,
    amr_val,
    context_probs_val,
    X_tst,
    amr_tst,
    context_probs_tst,
    corr_matrix,
    num_items,
    stage1_test_probs
):

    exp_name = experiment["name"]

    print(
        f"\nTraining Stage 2 | species={species} | fold={fold} | experiment={exp_name}",
        flush=True
    )

    loader_tr = make_stage2_loader(
        X=X_tr,
        amr=amr_tr,
        context_probs=context_probs_tr,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    loader_val = make_stage2_loader(
        X=X_val,
        amr=amr_val,
        context_probs=context_probs_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    model = Stage2ContextPredictor(
        num_feat=X_tr.shape[1],
        num_items=num_items,
        context_mode=experiment["context_mode"],
        low_threshold=experiment["low_threshold"],
        high_threshold=experiment["high_threshold"],
        top_k=experiment["top_k"],
        corr_kind=experiment["corr_kind"],
        corr_matrix=corr_matrix,
        drug_emb_dim=DRUG_EMB_DIM,
        maldi_hidden_dims=MALDI_HIDDEN_DIMS,
        maldi_emb_dim=MALDI_EMB_DIM,
        hidden_dims=STAGE2_HIDDEN_DIMS,
        lr=LR_STAGE2
    )

    callbacks = [
        EarlyStopping(
            monitor="loss_val",
            patience=PATIENCE_STAGE2,
            mode="min"
        )
    ]

    checkpoint_callback = None

    if USE_CHECKPOINT:
        checkpoint_callback = ModelCheckpoint(
            monitor="loss_val",
            mode="min",
            save_top_k=1,
            filename=(
                f"{safe_name(species)}_{exp_name}_fold{fold}"
                + "-{epoch:02d}-{loss_val:.4f}"
            )
        )

        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS_STAGE2,
        accelerator="gpu" if DEVICE == "cuda" else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=USE_CHECKPOINT
    )

    trainer.fit(model, loader_tr, loader_val)

    if USE_CHECKPOINT and checkpoint_callback is not None:
        best_path = checkpoint_callback.best_model_path

        if best_path is not None and best_path != "":
            print(f"Loading best Stage 2 checkpoint: {best_path}", flush=True)
            model = Stage2ContextPredictor.load_from_checkpoint(
                best_path,
                corr_matrix=corr_matrix
            )
            model = model.to(DEVICE)

    stage2_preds, context_masks = predict_stage2(
        model=model,
        X=X_tst,
        context_probs=context_probs_tst
    )

    stage2_micro, stage2_macro = compute_metrics(
        y_true=amr_tst,
        preds=stage2_preds
    )

    stage1_micro, stage1_macro = compute_metrics(
        y_true=amr_tst,
        preds=stage1_test_probs
    )

    coverage = compute_context_coverage(
        y_true=amr_tst,
        context_mask_per_target=context_masks
    )

    print(
        f"Result | {exp_name} | "
        f"stage2_micro={stage2_micro:.4f} | "
        f"stage2_macro={stage2_macro:.4f} | "
        f"stage1_micro={stage1_micro:.4f} | "
        f"stage1_macro={stage1_macro:.4f} | "
        f"coverage={coverage:.4f}",
        flush=True
    )

    return {
        "species": species,
        "fold": fold,
        "experiment": exp_name,
        "context_mode": experiment["context_mode"],
        "low_threshold": experiment["low_threshold"],
        "high_threshold": experiment["high_threshold"],
        "top_k": experiment["top_k"],
        "corr_kind": experiment["corr_kind"],
        "stage2_auc_micro": stage2_micro,
        "stage2_auc_macro": stage2_macro,
        "stage1_auc_micro": stage1_micro,
        "stage1_auc_macro": stage1_macro,
        "context_coverage": coverage,
        "calibration": "platt_per_antibiotic",
        "min_calibration_obs": MIN_CALIBRATION_OBS,
        "min_corr_pair": MIN_CORR_PAIR,
    }


# ============================================================
# TRAIN ONE SPECIES
# ============================================================
def train_species(species):

    print("\n====================", flush=True)
    print(f"Species: {species}", flush=True)
    print("====================", flush=True)

    data = build_species_data(species)

    if data is None:
        print("Skipping species...", flush=True)
        return []

    X, amr, antibiotic_names = data

    n_samples = X.shape[0]
    num_items = amr.shape[1]

    print(
        f"Samples: {n_samples} | Antibiotics: {num_items}",
        flush=True
    )

    folds = create_folds(
        n_samples=n_samples,
        n_splits=N_SPLITS
    )

    species_results = []

    for fold, (train_idx, test_idx) in enumerate(folds):

        print("\n--------------------", flush=True)
        print(f"Fold {fold}", flush=True)
        print("--------------------", flush=True)

        train_inner_idx, val_idx = split_train_val_indices(
            train_idx=train_idx,
            val_size=VAL_SIZE,
            seed=42 + fold
        )

        X_tr = X[train_inner_idx]
        X_val = X[val_idx]
        X_tst = X[test_idx]

        amr_tr = amr[train_inner_idx]
        amr_val = amr[val_idx]
        amr_tst = amr[test_idx]

        # ====================================================
        # 1) Train Stage 1
        # ====================================================
        stage1_model = train_stage1(
            species=species,
            fold=fold,
            X_tr=X_tr,
            amr_tr=amr_tr,
            X_val=X_val,
            amr_val=amr_val,
            num_items=num_items
        )

        # ====================================================
        # 2) Predict raw logits from Stage 1
        # ====================================================
        logits_tr = predict_stage1_logits(stage1_model, X_tr)
        logits_val = predict_stage1_logits(stage1_model, X_val)
        logits_tst = predict_stage1_logits(stage1_model, X_tst)

        raw_probs_tst = sigmoid_np(logits_tst)

        raw_stage1_micro, raw_stage1_macro = compute_metrics(
            y_true=amr_tst,
            preds=raw_probs_tst
        )

        print(
            f"Raw Stage 1 test | "
            f"micro={raw_stage1_micro:.4f} | "
            f"macro={raw_stage1_macro:.4f}",
            flush=True
        )

        # ====================================================
        # 3) Fit calibration on validation
        # ====================================================
        calibrator = PerAntibioticPlattCalibrator(
            min_obs=MIN_CALIBRATION_OBS
        )

        calibrator.fit(
            logits_val=logits_val,
            y_val=amr_val
        )

        df_cal = calibrator.to_dataframe(
            antibiotic_names=antibiotic_names
        )

        cal_path = os.path.join(
            OUTPUT_DIR,
            f"{safe_name(species)}_fold{fold}_platt_calibration.csv"
        )

        df_cal.to_csv(cal_path, index=False)

        print(
            f"Calibration fitted for "
            f"{df_cal['was_fitted'].sum()}/{num_items} antibiotics",
            flush=True
        )

        # ====================================================
        # 4) Apply calibration
        # ====================================================
        context_probs_tr = calibrator.transform(logits_tr)
        context_probs_val = calibrator.transform(logits_val)
        context_probs_tst = calibrator.transform(logits_tst)

        calibrated_stage1_micro, calibrated_stage1_macro = compute_metrics(
            y_true=amr_tst,
            preds=context_probs_tst
        )

        print(
            f"Calibrated Stage 1 test | "
            f"micro={calibrated_stage1_micro:.4f} | "
            f"macro={calibrated_stage1_macro:.4f}",
            flush=True
        )

        # ====================================================
        # 5) Compute train correlations
        # ====================================================
        corr_matrix, n_pair_matrix = compute_phi_correlation_matrix(
            amr_train=amr_tr,
            min_pair=MIN_CORR_PAIR
        )

        corr_path = os.path.join(
            OUTPUT_DIR,
            f"{safe_name(species)}_fold{fold}_train_phi_corr.csv"
        )

        pair_path = os.path.join(
            OUTPUT_DIR,
            f"{safe_name(species)}_fold{fold}_train_pair_counts.csv"
        )

        pd.DataFrame(
            corr_matrix,
            index=antibiotic_names,
            columns=antibiotic_names
        ).to_csv(corr_path)

        pd.DataFrame(
            n_pair_matrix,
            index=antibiotic_names,
            columns=antibiotic_names
        ).to_csv(pair_path)

        # ====================================================
        # 6) Train Stage 2 experiments
        # ====================================================
        for experiment in EXPERIMENTS:

            try:
                res = train_stage2(
                    experiment=experiment,
                    species=species,
                    fold=fold,
                    X_tr=X_tr,
                    amr_tr=amr_tr,
                    context_probs_tr=context_probs_tr,
                    X_val=X_val,
                    amr_val=amr_val,
                    context_probs_val=context_probs_val,
                    X_tst=X_tst,
                    amr_tst=amr_tst,
                    context_probs_tst=context_probs_tst,
                    corr_matrix=corr_matrix,
                    num_items=num_items,
                    stage1_test_probs=context_probs_tst
                )

                res["raw_stage1_auc_micro"] = raw_stage1_micro
                res["raw_stage1_auc_macro"] = raw_stage1_macro
                res["calibrated_stage1_auc_micro"] = calibrated_stage1_micro
                res["calibrated_stage1_auc_macro"] = calibrated_stage1_macro

                species_results.append(res)

                partial_species_path = os.path.join(
                    OUTPUT_DIR,
                    f"{safe_name(species)}_partial_results.csv"
                )

                pd.DataFrame(species_results).to_csv(
                    partial_species_path,
                    index=False
                )

            except Exception as e:
                print(
                    f"Error | species={species} | fold={fold} | "
                    f"experiment={experiment['name']} | {e}",
                    flush=True
                )

    return species_results


# ============================================================
# MAIN
# ============================================================
all_results = []

for sp in species_list:

    try:
        sp_results = train_species(sp)

        all_results.extend(sp_results)

        partial_path = os.path.join(
            OUTPUT_DIR,
            "all_calibrated_two_stage_partial_results.csv"
        )

        pd.DataFrame(all_results).to_csv(partial_path, index=False)

    except Exception as e:
        print(f"Error in species {sp}: {e}", flush=True)


df_raw = pd.DataFrame(all_results)

raw_path = os.path.join(
    OUTPUT_DIR,
    "calibrated_two_stage_raw_fold_results.csv"
)

df_raw.to_csv(raw_path, index=False)

print("\nRAW RESULTS:")
print(df_raw)
print(f"\nSaved raw results: {raw_path}")


# ============================================================
# SUMMARY BY SPECIES
# ============================================================
if len(df_raw) > 0:

    summary_species = (
        df_raw
        .groupby(["experiment", "species"], as_index=False)
        .agg(
            stage2_auc_micro=("stage2_auc_micro", "mean"),
            stage2_auc_macro=("stage2_auc_macro", "mean"),
            stage2_auc_micro_std=("stage2_auc_micro", "std"),
            stage2_auc_macro_std=("stage2_auc_macro", "std"),
            stage1_auc_micro=("stage1_auc_micro", "mean"),
            stage1_auc_macro=("stage1_auc_macro", "mean"),
            context_coverage=("context_coverage", "mean"),
            raw_stage1_auc_micro=("raw_stage1_auc_micro", "mean"),
            raw_stage1_auc_macro=("raw_stage1_auc_macro", "mean"),
            calibrated_stage1_auc_micro=("calibrated_stage1_auc_micro", "mean"),
            calibrated_stage1_auc_macro=("calibrated_stage1_auc_macro", "mean"),
        )
    )

    summary_species = summary_species.sort_values(
        by=["experiment", "stage2_auc_macro"],
        ascending=[True, False]
    )

    summary_species_path = os.path.join(
        OUTPUT_DIR,
        "calibrated_two_stage_summary_by_species.csv"
    )

    summary_species.to_csv(summary_species_path, index=False)

    print("\nSUMMARY BY SPECIES:")
    print(summary_species)
    print(f"\nSaved summary by species: {summary_species_path}")


    # ========================================================
    # GLOBAL SUMMARY BY EXPERIMENT
    # ========================================================
    summary_global = (
        summary_species
        .groupby("experiment", as_index=False)
        .agg(
            mean_stage2_auc_micro=("stage2_auc_micro", "mean"),
            mean_stage2_auc_macro=("stage2_auc_macro", "mean"),
            mean_stage1_auc_micro=("stage1_auc_micro", "mean"),
            mean_stage1_auc_macro=("stage1_auc_macro", "mean"),
            mean_context_coverage=("context_coverage", "mean"),
            mean_raw_stage1_auc_micro=("raw_stage1_auc_micro", "mean"),
            mean_raw_stage1_auc_macro=("raw_stage1_auc_macro", "mean"),
            mean_calibrated_stage1_auc_micro=("calibrated_stage1_auc_micro", "mean"),
            mean_calibrated_stage1_auc_macro=("calibrated_stage1_auc_macro", "mean"),
        )
    )

    summary_global = summary_global.sort_values(
        by="mean_stage2_auc_macro",
        ascending=False
    )

    summary_global_path = os.path.join(
        OUTPUT_DIR,
        "calibrated_two_stage_global_summary.csv"
    )

    summary_global.to_csv(summary_global_path, index=False)

    print("\nGLOBAL SUMMARY BY EXPERIMENT:")
    print(summary_global)
    print(f"\nSaved global summary: {summary_global_path}")


print("\nDone.")