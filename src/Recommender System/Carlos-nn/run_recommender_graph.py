import os
import pickle
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from pytorch_lightning.callbacks import EarlyStopping

from lib.NCF import NCF
from lib.RecDataset import RecDataset


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl"

OUTPUT_DIR = "graph_recommender_experiment_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42

N_SPLITS = 2
VAL_FRACTION = 0.2

# Base NCF config
BASE_BATCH_SIZE = 64
BASE_TRAIN_BATCH_SIZE = 128
BASE_MAX_EPOCHS = 300
BASE_PATIENCE = 15

EMPTY_CONTEXT_PROB = 0.05
MIN_CONTEXT_FRACTION = 0.5
MAX_CONTEXT_FRACTION = 1.0

# Graph config
GRAPH_BATCH_SIZE = 64
GRAPH_MAX_EPOCHS = 300
GRAPH_PATIENCE = 25
GRAPH_LR = 1e-3
GRAPH_WEIGHT_DECAY = 1e-5

GRAPH_HIDDEN_DIM = 128
GRAPH_NUM_LAYERS = 2
DRUG_EMB_DIM_GRAPH = 32
MALDI_EMB_DIM_GRAPH = 64

# Adjacency / graph construction
MIN_CORR_PAIR = 30
MIN_ABS_CORR = 0.05
USE_SIGNED_CORRELATIONS = True

# If True, A starts from the correlation graph but is fine-tuned during training
OPTION_A_TRAINABLE_ADJ = True
OPTION_B_TRAINABLE_ADJ = True

# Penalizes the trainable graph if it moves too far from the initial graph
ADJ_REG_LAMBDA = 1e-3

NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("Using device:", DEVICE, flush=True)
print("Loading dataset...", flush=True)

with open(DATA_PATH, "rb") as f:
    payload = pickle.load(f)

X_all = payload["data"]
y_species_all = payload["label"]
amr_all = payload["amr"]
antibiotics_all = payload["antibiotics"]

species_list = np.unique(y_species_all)

print("Total samples:", len(X_all), flush=True)
print("Total species:", len(species_list), flush=True)


# ============================================================
# DATA PREPARATION
# ============================================================

def build_species_data(species):
    """
    Builds species-specific MALDI matrix and AMR matrix.

    Keeps only antibiotics with:
        - more than 50 observed values
        - both classes present
    """

    mask = y_species_all == species

    X = X_all[mask]
    amr = amr_all[mask].copy()

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

    X = np.asarray(X, dtype=np.float32)
    amr = amr[:, valid_cols].astype(np.float32)

    antibiotics = np.asarray(antibiotics_all)[valid_cols]

    return X, amr, antibiotics


def create_folds(n_samples, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(kf.split(np.arange(n_samples)))


def split_train_val(train_idx, val_fraction=0.2, seed=42):
    """
    Splits the training fold into train and validation.

    This avoids using the test fold as validation.
    """

    train_idx = np.asarray(train_idx)

    if len(train_idx) < 10:
        return train_idx, train_idx

    tr_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True
    )

    return np.asarray(tr_idx), np.asarray(val_idx)


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def compute_phi_correlation_matrix(amr_train, min_pair=30):
    """
    Computes phi correlation between antibiotic resistance profiles.

    Only training data must be used.

    amr_train:
        shape = [num_samples, num_antibiotics]
        values = 0, 1, NaN
    """

    num_items = amr_train.shape[1]
    phi = np.zeros((num_items, num_items), dtype=np.float32)

    for i in range(num_items):
        for j in range(i + 1, num_items):

            xi = amr_train[:, i]
            xj = amr_train[:, j]

            valid = ~np.isnan(xi) & ~np.isnan(xj)

            if np.sum(valid) < min_pair:
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
                continue

            value = ((n11 * n00) - (n10 * n01)) / denom

            phi[i, j] = value
            phi[j, i] = value

    return phi


def build_normalized_adjacency(
    phi,
    min_abs_corr=0.05,
    use_signed=True,
    add_self_loops=True
):
    """
    Builds normalized adjacency matrix from phi correlations.

    If use_signed=True:
        positive and negative correlations are kept.

    If use_signed=False:
        absolute correlations are used.

    Normalization uses absolute degree:
        A_norm[i, j] = A[i, j] / sqrt(deg_i * deg_j)
    where:
        deg_i = sum_j |A[i, j]|
    """

    A = phi.copy().astype(np.float32)

    if not use_signed:
        A = np.abs(A)

    A[np.abs(A) < min_abs_corr] = 0.0

    if add_self_loops:
        np.fill_diagonal(A, 1.0)

    deg = np.sum(np.abs(A), axis=1)
    deg[deg == 0] = 1.0

    D_inv_sqrt = 1.0 / np.sqrt(deg)
    A_norm = D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :]

    return A_norm.astype(np.float32)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, preds):
    """
    Computes micro and macro AUC with missing labels ignored.
    """

    y_true = np.asarray(y_true)
    preds = np.asarray(preds)

    mask = ~np.isnan(y_true)

    auc_micro = np.nan
    auc_macro = np.nan

    if np.any(mask) and len(np.unique(y_true[mask])) > 1:
        auc_micro = roc_auc_score(y_true[mask], preds[mask])

    antibiotic_aucs = []

    for j in range(y_true.shape[1]):

        col = y_true[:, j]
        valid = ~np.isnan(col)

        if np.sum(valid) > 0 and len(np.unique(col[valid])) > 1:
            antibiotic_aucs.append(
                roc_auc_score(col[valid], preds[valid, j])
            )

    if len(antibiotic_aucs) > 0:
        auc_macro = float(np.mean(antibiotic_aucs))

    return auc_micro, auc_macro


# ============================================================
# BASE NCF PREDICTION UTILITIES
# ============================================================

def predict_all_antibiotics_ncf(
    model,
    maldi,
    amr_context,
    context_mask,
    num_items,
    batch_size=512
):
    """
    Predicts resistance probability for every antibiotic for every sample
    using the original NCF model.

    Output:
        preds_matrix: shape [N samples, num_items]
    """

    model.eval()

    N = maldi.shape[0]
    preds_matrix = np.zeros((N, num_items), dtype=np.float32)

    device = model.device

    with torch.no_grad():

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            maldi_batch = torch.tensor(maldi[start:end]).float().to(device)
            amr_batch = torch.tensor(amr_context[start:end]).float().to(device)
            mask_batch = torch.tensor(context_mask[start:end]).float().to(device)

            batch_size_real = maldi_batch.shape[0]

            drug_preds = []

            for drug_id in range(num_items):

                target_drug = torch.full(
                    (batch_size_real,),
                    drug_id,
                    dtype=torch.long,
                    device=device
                )

                target_family = torch.zeros(
                    batch_size_real,
                    dtype=torch.long,
                    device=device
                )

                amr_tmp = amr_batch.clone()
                mask_tmp = mask_batch.clone()

                amr_tmp[:, drug_id] = 0.0
                mask_tmp[:, drug_id] = 0.0

                preds = model(
                    maldi_batch,
                    target_drug,
                    target_family,
                    amr_tmp,
                    mask_tmp
                )

                drug_preds.append(preds.view(-1))

            drug_preds = torch.stack(drug_preds, dim=1)

            preds_matrix[start:end] = drug_preds.cpu().numpy()

    return preds_matrix


def one_shot_inference_ncf(model, maldi, num_items):
    """
    NCF one-shot inference:
        all AMR context masked.
    """

    N = maldi.shape[0]

    amr_context = np.zeros((N, num_items), dtype=np.float32)
    context_mask = np.zeros((N, num_items), dtype=np.float32)

    preds = predict_all_antibiotics_ncf(
        model=model,
        maldi=maldi,
        amr_context=amr_context,
        context_mask=context_mask,
        num_items=num_items
    )

    return preds


# ============================================================
# DATASETS FOR GRAPH MODELS
# ============================================================

class GraphRefinerDataset(Dataset):
    """
    Dataset for Option A.

    Input:
        raw_probs: base NCF one-shot probabilities, shape [N, num_antibiotics]

    Target:
        AMR matrix, shape [N, num_antibiotics]

    Mask:
        1 where AMR label exists, 0 where NaN
    """

    def __init__(self, raw_probs, amr):
        self.raw_probs = np.asarray(raw_probs, dtype=np.float32)
        self.amr = np.asarray(amr, dtype=np.float32)
        self.mask = (~np.isnan(self.amr)).astype(np.float32)

        y = self.amr.copy()
        y[np.isnan(y)] = 0.0
        self.y = y.astype(np.float32)

    def __len__(self):
        return self.raw_probs.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.raw_probs[idx]).float(),
            torch.tensor(self.y[idx]).float(),
            torch.tensor(self.mask[idx]).float()
        )


class GraphIntegratedDataset(Dataset):
    """
    Dataset for Option B.

    Input:
        MALDI spectrum

    Target:
        Full AMR profile, with missing labels masked.
    """

    def __init__(self, maldi, amr):
        self.maldi = np.asarray(maldi, dtype=np.float32)
        self.amr = np.asarray(amr, dtype=np.float32)
        self.mask = (~np.isnan(self.amr)).astype(np.float32)

        y = self.amr.copy()
        y[np.isnan(y)] = 0.0
        self.y = y.astype(np.float32)

    def __len__(self):
        return self.maldi.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.maldi[idx]).float(),
            torch.tensor(self.y[idx]).float(),
            torch.tensor(self.mask[idx]).float()
        )


# ============================================================
# GRAPH MODULES
# ============================================================

class GraphMessagePassing(nn.Module):
    """
    Simple GCN-style message passing block.

    H shape:
        [batch, num_nodes, hidden_dim]

    A shape:
        [num_nodes, num_nodes]

    Update:
        H_new = GELU( W_self(H) + W_msg(A @ H) )
    """

    def __init__(
        self,
        adjacency_init,
        hidden_dim,
        num_layers=2,
        dropout=0.2,
        trainable_adj=False,
        adj_reg_lambda=1e-3
    ):
        super().__init__()

        A = torch.tensor(adjacency_init, dtype=torch.float32)

        self.register_buffer("A_init", A)
        self.register_buffer("edge_mask", (torch.abs(A) > 0).float())

        self.trainable_adj = trainable_adj
        self.adj_reg_lambda = adj_reg_lambda

        if trainable_adj:
            self.A_delta = nn.Parameter(torch.zeros_like(A))
        else:
            self.A_delta = None

        self.self_linears = nn.ModuleList()
        self.msg_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(num_layers):
            self.self_linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.msg_linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))

        self.act = nn.GELU()

    def get_adjacency(self):
        if self.trainable_adj:
            delta = 0.5 * (self.A_delta + self.A_delta.T)
            A = self.A_init + delta * self.edge_mask
        else:
            A = self.A_init

        # Re-normalize by absolute degree for stability
        deg = torch.sum(torch.abs(A), dim=1)
        deg = torch.clamp(deg, min=1e-6)
        D_inv_sqrt = torch.rsqrt(deg)

        A_norm = D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :]

        return A_norm

    def adjacency_regularization(self):
        if not self.trainable_adj:
            return torch.tensor(0.0, device=self.A_init.device)

        A_current = self.get_adjacency()
        return self.adj_reg_lambda * torch.mean((A_current - self.A_init) ** 2)

    def forward(self, H):
        A = self.get_adjacency()

        for self_lin, msg_lin, norm, dropout in zip(
            self.self_linears,
            self.msg_linears,
            self.norms,
            self.dropouts
        ):
            # A @ H over the node dimension
            H_msg = torch.einsum("ij,bjd->bid", A, H)

            H_new = self_lin(H) + msg_lin(H_msg)
            H_new = self.act(H_new)
            H_new = dropout(H_new)
            H = norm(H + H_new)

        return H


# ============================================================
# OPTION A MODEL
# ============================================================

class OptionA_GraphLogitRefiner(nn.Module):
    """
    Option A:

        Base NCF one-shot probabilities
                ↓
        convert to logits
                ↓
        node features = [raw_logit_i, antibiotic_embedding_i]
                ↓
        GNN over antibiotic graph
                ↓
        refined logits per antibiotic
    """

    def __init__(
        self,
        num_antibiotics,
        adjacency_init,
        drug_emb_dim=32,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        trainable_adj=True,
        adj_reg_lambda=1e-3
    ):
        super().__init__()

        self.num_antibiotics = num_antibiotics

        self.drug_embedding = nn.Embedding(num_antibiotics, drug_emb_dim)

        self.node_init = nn.Sequential(
            nn.Linear(1 + drug_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        self.gnn = GraphMessagePassing(
            adjacency_init=adjacency_init,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            trainable_adj=trainable_adj,
            adj_reg_lambda=adj_reg_lambda
        )

        self.out = nn.Linear(hidden_dim, 1)

    @staticmethod
    def prob_to_logit(p):
        eps = 1e-6
        p = torch.clamp(p, eps, 1.0 - eps)
        return torch.log(p / (1.0 - p))

    def forward(self, raw_probs):
        """
        raw_probs:
            [batch, num_antibiotics]

        returns:
            logits [batch, num_antibiotics]
        """

        B, N = raw_probs.shape
        device = raw_probs.device

        raw_logits = self.prob_to_logit(raw_probs).unsqueeze(-1)

        drug_ids = torch.arange(N, device=device)
        drug_emb = self.drug_embedding(drug_ids)
        drug_emb = drug_emb.unsqueeze(0).expand(B, N, -1)

        X = torch.cat([raw_logits, drug_emb], dim=-1)

        H = self.node_init(X)
        H = self.gnn(H)

        logits = self.out(H).squeeze(-1)

        return logits

    def regularization_loss(self):
        return self.gnn.adjacency_regularization()


# ============================================================
# OPTION B MODEL
# ============================================================

class EmSpectrumGraph(nn.Module):
    """
    MALDI encoder for Option B.
    Same spirit as your NCF Em_Spectrum.
    """

    def __init__(self, input_dim, output_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x.float())


class OptionB_IntegratedGraphRecommender(nn.Module):
    """
    Option B:

        MALDI spectrum
              ↓
        MALDI embedding z_s
              ↓
        for each antibiotic:
            node_i = MLP([z_s, antibiotic_embedding_i])
              ↓
        GNN over antibiotic graph
              ↓
        logits per antibiotic
    """

    def __init__(
        self,
        num_features,
        num_antibiotics,
        adjacency_init,
        maldi_emb_dim=64,
        drug_emb_dim=32,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        trainable_adj=True,
        adj_reg_lambda=1e-3
    ):
        super().__init__()

        self.num_antibiotics = num_antibiotics

        self.maldi_encoder = EmSpectrumGraph(
            input_dim=num_features,
            output_dim=maldi_emb_dim
        )

        self.drug_embedding = nn.Embedding(num_antibiotics, drug_emb_dim)

        self.node_init = nn.Sequential(
            nn.Linear(maldi_emb_dim + drug_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        self.gnn = GraphMessagePassing(
            adjacency_init=adjacency_init,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            trainable_adj=trainable_adj,
            adj_reg_lambda=adj_reg_lambda
        )

        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, maldi):
        """
        maldi:
            [batch, num_features]

        returns:
            logits [batch, num_antibiotics]
        """

        B = maldi.shape[0]
        N = self.num_antibiotics
        device = maldi.device

        z = self.maldi_encoder(maldi)
        z = z.unsqueeze(1).expand(B, N, -1)

        drug_ids = torch.arange(N, device=device)
        drug_emb = self.drug_embedding(drug_ids)
        drug_emb = drug_emb.unsqueeze(0).expand(B, N, -1)

        X = torch.cat([z, drug_emb], dim=-1)

        H = self.node_init(X)
        H = self.gnn(H)

        logits = self.out(H).squeeze(-1)

        return logits

    def regularization_loss(self):
        return self.gnn.adjacency_regularization()


# ============================================================
# TRAINING / EVALUATION FOR GRAPH MODELS
# ============================================================

def masked_bce_with_logits(logits, labels, mask):
    """
    BCEWithLogits over observed labels only.
    """

    loss_raw = nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none"
    )

    loss = loss_raw * mask

    denom = torch.clamp(mask.sum(), min=1.0)

    return loss.sum() / denom


def train_option_a_graph_refiner(
    model,
    train_loader,
    val_loader,
    max_epochs=300,
    patience=25,
    lr=1e-3,
    weight_decay=1e-5,
    device="cuda"
):
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_val = np.inf
    best_state = None
    bad_epochs = 0

    for epoch in range(max_epochs):

        model.train()
        train_losses = []

        for raw_probs, labels, mask in train_loader:
            raw_probs = raw_probs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            logits = model(raw_probs)

            loss = masked_bce_with_logits(logits, labels, mask)
            loss = loss + model.regularization_loss()

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for raw_probs, labels, mask in val_loader:
                raw_probs = raw_probs.to(device)
                labels = labels.to(device)
                mask = mask.to(device)

                logits = model(raw_probs)

                loss = masked_bce_with_logits(logits, labels, mask)
                loss = loss + model.regularization_loss()

                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_loss = float(np.mean(val_losses)) if val_losses else np.nan

        print(
            f"    Option A epoch {epoch + 1:03d} | "
            f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f}",
            flush=True
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"    Option A early stopping at epoch {epoch + 1}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def train_option_b_integrated_graph(
    model,
    train_loader,
    val_loader,
    max_epochs=300,
    patience=25,
    lr=1e-3,
    weight_decay=1e-5,
    device="cuda"
):
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_val = np.inf
    best_state = None
    bad_epochs = 0

    for epoch in range(max_epochs):

        model.train()
        train_losses = []

        for maldi, labels, mask in train_loader:
            maldi = maldi.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            logits = model(maldi)

            loss = masked_bce_with_logits(logits, labels, mask)
            loss = loss + model.regularization_loss()

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for maldi, labels, mask in val_loader:
                maldi = maldi.to(device)
                labels = labels.to(device)
                mask = mask.to(device)

                logits = model(maldi)

                loss = masked_bce_with_logits(logits, labels, mask)
                loss = loss + model.regularization_loss()

                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_loss = float(np.mean(val_losses)) if val_losses else np.nan

        print(
            f"    Option B epoch {epoch + 1:03d} | "
            f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f}",
            flush=True
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"    Option B early stopping at epoch {epoch + 1}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict_option_a(model, raw_probs, batch_size=512, device="cuda"):
    model.eval()
    model = model.to(device)

    preds = []

    with torch.no_grad():
        for start in range(0, raw_probs.shape[0], batch_size):
            end = min(start + batch_size, raw_probs.shape[0])

            x = torch.tensor(raw_probs[start:end]).float().to(device)

            logits = model(x)
            prob = torch.sigmoid(logits)

            preds.append(prob.cpu().numpy())

    return np.concatenate(preds, axis=0)


def predict_option_b(model, maldi, batch_size=512, device="cuda"):
    model.eval()
    model = model.to(device)

    preds = []

    with torch.no_grad():
        for start in range(0, maldi.shape[0], batch_size):
            end = min(start + batch_size, maldi.shape[0])

            x = torch.tensor(maldi[start:end]).float().to(device)

            logits = model(x)
            prob = torch.sigmoid(logits)

            preds.append(prob.cpu().numpy())

    return np.concatenate(preds, axis=0)


# ============================================================
# BASE NCF TRAINING
# ============================================================

def train_base_ncf(
    X_tr,
    amr_tr,
    X_val,
    amr_val,
    num_items
):
    """
    Trains your original NCF model using RecDataset.
    """

    antibiotic_families = np.zeros(num_items, dtype=int)
    num_families = 1

    train_sample_ids = np.arange(len(X_tr))
    val_sample_ids = np.arange(len(X_val))

    train_sample_amr = {
        i: amr_tr[i]
        for i in range(len(X_tr))
    }

    val_sample_amr = {
        i: amr_val[i]
        for i in range(len(X_val))
    }

    loader_tr = DataLoader(
        RecDataset(
            maldi=X_tr,
            sample_ids=train_sample_ids,
            sample_amr=train_sample_amr,
            antibiotic_families=antibiotic_families,
            empty_context_prob=EMPTY_CONTEXT_PROB,
            min_context_fraction=MIN_CONTEXT_FRACTION,
            max_context_fraction=MAX_CONTEXT_FRACTION
        ),
        batch_size=BASE_TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    loader_val = DataLoader(
        RecDataset(
            maldi=X_val,
            sample_ids=val_sample_ids,
            sample_amr=val_sample_amr,
            antibiotic_families=antibiotic_families,
            empty_context_prob=EMPTY_CONTEXT_PROB,
            min_context_fraction=MIN_CONTEXT_FRACTION,
            max_context_fraction=MAX_CONTEXT_FRACTION
        ),
        batch_size=BASE_BATCH_SIZE,
        shuffle=False,
        num_workers=max(1, NUM_WORKERS // 2),
        pin_memory=True
    )

    model = NCF(
        num_feat=X_tr.shape[1],
        num_items=num_items,
        num_families=num_families,
        amr_dropout=0.1
    )

    trainer = pl.Trainer(
        max_epochs=BASE_MAX_EPOCHS,
        accelerator="gpu" if DEVICE == "cuda" else "cpu",
        devices=1,
        callbacks=[
            EarlyStopping(
                monitor="loss_val",
                patience=BASE_PATIENCE,
                mode="min"
            )
        ],
        logger=False,
        enable_checkpointing=False
    )

    trainer.fit(model, loader_tr, loader_val)

    return model


# ============================================================
# TRAIN / EVALUATE ONE SPECIES
# ============================================================

def train_species(species):
    print("\n" + "=" * 70, flush=True)
    print(f"Species: {species}", flush=True)
    print("=" * 70, flush=True)

    data = build_species_data(species)

    if data is None:
        print("Skipping species: not enough usable data.", flush=True)
        return []

    X, amr, antibiotics = data

    num_items = amr.shape[1]
    n_samples = X.shape[0]

    print("Samples:", n_samples, "| Antibiotics:", num_items, flush=True)
    print("Antibiotics used:", list(antibiotics), flush=True)

    folds = create_folds(n_samples, N_SPLITS)

    species_results = []

    for fold, (train_val_idx, test_idx) in enumerate(folds):
        print("\n" + "-" * 70, flush=True)
        print(f"Fold {fold}", flush=True)
        print("-" * 70, flush=True)

        train_idx, val_idx = split_train_val(
            train_val_idx,
            val_fraction=VAL_FRACTION,
            seed=SEED + fold
        )

        X_tr = X[train_idx]
        X_val = X[val_idx]
        X_tst = X[test_idx]

        amr_tr = amr[train_idx]
        amr_val = amr[val_idx]
        amr_tst = amr[test_idx]

        print(
            f"Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_tst)}",
            flush=True
        )

        # ------------------------------------------------------------
        # Build graph from TRAIN ONLY
        # ------------------------------------------------------------
        phi_train = compute_phi_correlation_matrix(
            amr_train=amr_tr,
            min_pair=MIN_CORR_PAIR
        )

        adjacency = build_normalized_adjacency(
            phi=phi_train,
            min_abs_corr=MIN_ABS_CORR,
            use_signed=USE_SIGNED_CORRELATIONS,
            add_self_loops=True
        )

        print(
            "Adjacency stats:",
            "min =", float(np.min(adjacency)),
            "mean =", float(np.mean(adjacency)),
            "max =", float(np.max(adjacency)),
            "nonzero =", int(np.sum(np.abs(adjacency) > 0)),
            "/",
            adjacency.size,
            flush=True
        )

        # ------------------------------------------------------------
        # Train baseline NCF
        # ------------------------------------------------------------
        print("\nTraining baseline NCF...", flush=True)

        base_model = train_base_ncf(
            X_tr=X_tr,
            amr_tr=amr_tr,
            X_val=X_val,
            amr_val=amr_val,
            num_items=num_items
        )

        # ------------------------------------------------------------
        # Baseline NCF one-shot predictions
        # ------------------------------------------------------------
        print("\nComputing NCF one-shot predictions...", flush=True)

        raw_tr = one_shot_inference_ncf(
            model=base_model,
            maldi=X_tr,
            num_items=num_items
        )

        raw_val = one_shot_inference_ncf(
            model=base_model,
            maldi=X_val,
            num_items=num_items
        )

        raw_tst = one_shot_inference_ncf(
            model=base_model,
            maldi=X_tst,
            num_items=num_items
        )

        base_micro, base_macro = compute_metrics(amr_tst, raw_tst)

        print(
            f"Baseline NCF one-shot | micro AUC={base_micro:.4f} | macro AUC={base_macro:.4f}",
            flush=True
        )

        species_results.append({
            "species": species,
            "fold": fold,
            "model": "baseline_ncf_one_shot",
            "auc_micro": base_micro,
            "auc_macro": base_macro,
            "num_samples": n_samples,
            "num_antibiotics": num_items
        })

        # ------------------------------------------------------------
        # Option A: NCF logits/probs + graph refiner
        # ------------------------------------------------------------
        print("\nTraining Option A: NCF one-shot -> graph logit refiner...", flush=True)

        loader_a_tr = DataLoader(
            GraphRefinerDataset(raw_tr, amr_tr),
            batch_size=GRAPH_BATCH_SIZE,
            shuffle=True,
            num_workers=max(1, NUM_WORKERS // 2),
            pin_memory=True
        )

        loader_a_val = DataLoader(
            GraphRefinerDataset(raw_val, amr_val),
            batch_size=GRAPH_BATCH_SIZE,
            shuffle=False,
            num_workers=max(1, NUM_WORKERS // 2),
            pin_memory=True
        )

        option_a_model = OptionA_GraphLogitRefiner(
            num_antibiotics=num_items,
            adjacency_init=adjacency,
            drug_emb_dim=DRUG_EMB_DIM_GRAPH,
            hidden_dim=GRAPH_HIDDEN_DIM,
            num_layers=GRAPH_NUM_LAYERS,
            dropout=0.2,
            trainable_adj=OPTION_A_TRAINABLE_ADJ,
            adj_reg_lambda=ADJ_REG_LAMBDA
        )

        option_a_model = train_option_a_graph_refiner(
            model=option_a_model,
            train_loader=loader_a_tr,
            val_loader=loader_a_val,
            max_epochs=GRAPH_MAX_EPOCHS,
            patience=GRAPH_PATIENCE,
            lr=GRAPH_LR,
            weight_decay=GRAPH_WEIGHT_DECAY,
            device=DEVICE
        )

        preds_a = predict_option_a(
            model=option_a_model,
            raw_probs=raw_tst,
            device=DEVICE
        )

        a_micro, a_macro = compute_metrics(amr_tst, preds_a)

        print(
            f"Option A | micro AUC={a_micro:.4f} | macro AUC={a_macro:.4f}",
            flush=True
        )

        species_results.append({
            "species": species,
            "fold": fold,
            "model": "option_a_ncf_plus_graph_refiner",
            "auc_micro": a_micro,
            "auc_macro": a_macro,
            "num_samples": n_samples,
            "num_antibiotics": num_items
        })

        # ------------------------------------------------------------
        # Option B: integrated MALDI + antibiotic embeddings + graph
        # ------------------------------------------------------------
        print("\nTraining Option B: integrated MALDI-antibiotic graph recommender...", flush=True)

        loader_b_tr = DataLoader(
            GraphIntegratedDataset(X_tr, amr_tr),
            batch_size=GRAPH_BATCH_SIZE,
            shuffle=True,
            num_workers=max(1, NUM_WORKERS // 2),
            pin_memory=True
        )

        loader_b_val = DataLoader(
            GraphIntegratedDataset(X_val, amr_val),
            batch_size=GRAPH_BATCH_SIZE,
            shuffle=False,
            num_workers=max(1, NUM_WORKERS // 2),
            pin_memory=True
        )

        option_b_model = OptionB_IntegratedGraphRecommender(
            num_features=X_tr.shape[1],
            num_antibiotics=num_items,
            adjacency_init=adjacency,
            maldi_emb_dim=MALDI_EMB_DIM_GRAPH,
            drug_emb_dim=DRUG_EMB_DIM_GRAPH,
            hidden_dim=GRAPH_HIDDEN_DIM,
            num_layers=GRAPH_NUM_LAYERS,
            dropout=0.2,
            trainable_adj=OPTION_B_TRAINABLE_ADJ,
            adj_reg_lambda=ADJ_REG_LAMBDA
        )

        option_b_model = train_option_b_integrated_graph(
            model=option_b_model,
            train_loader=loader_b_tr,
            val_loader=loader_b_val,
            max_epochs=GRAPH_MAX_EPOCHS,
            patience=GRAPH_PATIENCE,
            lr=GRAPH_LR,
            weight_decay=GRAPH_WEIGHT_DECAY,
            device=DEVICE
        )

        preds_b = predict_option_b(
            model=option_b_model,
            maldi=X_tst,
            device=DEVICE
        )

        b_micro, b_macro = compute_metrics(amr_tst, preds_b)

        print(
            f"Option B | micro AUC={b_micro:.4f} | macro AUC={b_macro:.4f}",
            flush=True
        )

        species_results.append({
            "species": species,
            "fold": fold,
            "model": "option_b_integrated_graph_recommender",
            "auc_micro": b_micro,
            "auc_macro": b_macro,
            "num_samples": n_samples,
            "num_antibiotics": num_items
        })

        # ------------------------------------------------------------
        # Save fold-level predictions for later inspection
        # ------------------------------------------------------------
        fold_prefix = f"{species}_fold_{fold}".replace(" ", "_").replace("/", "_")

        np.save(
            os.path.join(OUTPUT_DIR, f"{fold_prefix}_baseline_preds.npy"),
            raw_tst
        )
        np.save(
            os.path.join(OUTPUT_DIR, f"{fold_prefix}_option_a_preds.npy"),
            preds_a
        )
        np.save(
            os.path.join(OUTPUT_DIR, f"{fold_prefix}_option_b_preds.npy"),
            preds_b
        )
        np.save(
            os.path.join(OUTPUT_DIR, f"{fold_prefix}_y_test.npy"),
            amr_tst
        )
        np.save(
            os.path.join(OUTPUT_DIR, f"{fold_prefix}_adjacency.npy"),
            adjacency
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

        df_partial = pd.DataFrame(all_results)
        df_partial.to_csv(
            os.path.join(OUTPUT_DIR, "graph_experiment_results_by_fold_partial.csv"),
            index=False
        )

    except Exception as e:
        print(f"\nERROR in species {sp}: {e}", flush=True)


df = pd.DataFrame(all_results)

if len(df) == 0:
    print("No results generated.", flush=True)
    raise SystemExit


# Fold-level results
df_fold = df.copy()
df_fold.to_csv(
    os.path.join(OUTPUT_DIR, "graph_experiment_results_by_fold.csv"),
    index=False
)


# Species-level summary
df_species_summary = (
    df.groupby(["species", "model"], as_index=False)
    .agg(
        auc_micro_mean=("auc_micro", "mean"),
        auc_micro_std=("auc_micro", "std"),
        auc_macro_mean=("auc_macro", "mean"),
        auc_macro_std=("auc_macro", "std"),
        n_folds=("fold", "count"),
        num_samples=("num_samples", "first"),
        num_antibiotics=("num_antibiotics", "first")
    )
)

df_species_summary = df_species_summary.sort_values(
    by=["species", "auc_macro_mean"],
    ascending=[True, False]
)

df_species_summary.to_csv(
    os.path.join(OUTPUT_DIR, "graph_experiment_species_summary.csv"),
    index=False
)


# Global model summary
df_global_summary = (
    df.groupby("model", as_index=False)
    .agg(
        auc_micro_mean=("auc_micro", "mean"),
        auc_micro_std=("auc_micro", "std"),
        auc_macro_mean=("auc_macro", "mean"),
        auc_macro_std=("auc_macro", "std"),
        n_entries=("auc_micro", "count")
    )
)

df_global_summary = df_global_summary.sort_values(
    by="auc_macro_mean",
    ascending=False
)

df_global_summary.to_csv(
    os.path.join(OUTPUT_DIR, "graph_experiment_global_summary.csv"),
    index=False
)


print("\n" + "=" * 70)
print("FINAL FOLD-LEVEL RESULTS")
print("=" * 70)
print(df_fold)

print("\n" + "=" * 70)
print("FINAL SPECIES SUMMARY")
print("=" * 70)
print(df_species_summary)

print("\n" + "=" * 70)
print("FINAL GLOBAL SUMMARY")
print("=" * 70)
print(df_global_summary)

print("\nSaved outputs in:", OUTPUT_DIR)
print("Files:")
print("  graph_experiment_results_by_fold.csv")
print("  graph_experiment_species_summary.csv")
print("  graph_experiment_global_summary.csv")
print("  plus .npy files with predictions, y_test and adjacency per species/fold")