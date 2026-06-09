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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl"

N_SPLITS = 2
VAL_SIZE = 0.2

BATCH_SIZE = 64
MAX_EPOCHS = 300
PATIENCE = 15
LR = 1e-3

DRUG_EMB_DIM = 32

# Más profundo que el anterior:
# si el MALDI tiene ~6000 features, evitamos saltar directamente 6000 -> 512.
MALDI_HIDDEN_DIMS = [2048, 1024, 512, 256, 128]
MALDI_EMB_DIM = 128

STAGE1_HIDDEN_DIMS = [256, 128, 64]
STAGE2_HIDDEN_DIMS = [256, 128, 64]

# Peso de la loss auxiliar de la primera MLP:
# MALDI + antibiotic embedding -> pseudo-antibiograma.
LAMBDA_STAGE1 = 0.3

USE_CHECKPOINT = True

OUTPUT_DIR = "two_stage_context_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE, flush=True)

torch.backends.cudnn.benchmark = True


# ============================================================
# EXPERIMENTS
# ============================================================
# 1) selective 10/90:
#    p <= 0.10 -> contexto visible como 0
#    p >= 0.90 -> contexto visible como 1
#    el resto -> missing
#
# 2) selective 10/80:
#    p <= 0.10 -> contexto visible como 0
#    p >= 0.80 -> contexto visible como 1
#    el resto -> missing
#
# Si en realidad quieres 20/80, cambia low=0.20.
#
# 3) top-k:
#    se quedan solo los antibióticos más confiados según |p - 0.5|.
#
# 4) oracle_50:
#    techo del pseudo-contexto.
#    Usa el 50% de resistencias reales del test como contexto,
#    excluyendo siempre el antibiótico target.
EXPERIMENTS = [
    {
        "name": "selective_10_90",
        "context_mode": "threshold",
        "low_threshold": 0.10,
        "high_threshold": 0.90,
        "top_k": None,
        "oracle_fraction": None,
        "oracle_noise": 0.0,
    },
    {
        "name": "selective_10_80",
        "context_mode": "threshold",
        "low_threshold": 0.10,
        "high_threshold": 0.80,
        "top_k": None,
        "oracle_fraction": None,
        "oracle_noise": 0.0,
    },
    {
        "name": "topk_3",
        "context_mode": "topk",
        "low_threshold": None,
        "high_threshold": None,
        "top_k": 3,
        "oracle_fraction": None,
        "oracle_noise": 0.0,
    },
    {
        "name": "topk_5",
        "context_mode": "topk",
        "low_threshold": None,
        "high_threshold": None,
        "top_k": 5,
        "oracle_fraction": None,
        "oracle_noise": 0.0,
    },
    {
        "name": "oracle_50",
        "context_mode": "oracle",
        "low_threshold": None,
        "high_threshold": None,
        "top_k": None,
        "oracle_fraction": 0.50,
        "oracle_noise": 0.0,
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
# DATASET
# ============================================================
class AMRVectorDataset(Dataset):
    """
    Cada item es una muestra completa.

    Devuelve:
        maldi:  espectro MALDI
        labels: vector AMR con NaN -> 0
        mask:   1 donde hay AMR real, 0 donde era NaN
    """

    def __init__(self, maldi, amr):
        self.maldi = np.asarray(maldi)
        self.amr = np.asarray(amr)

        if self.maldi.shape[0] != self.amr.shape[0]:
            raise ValueError(
                "maldi and amr must have the same number of samples. "
                f"Got maldi={self.maldi.shape[0]}, amr={self.amr.shape[0]}"
            )

        valid_values = (
            (self.amr == 0) |
            (self.amr == 1) |
            np.isnan(self.amr)
        )

        if not np.all(valid_values):
            raise ValueError(
                "AMR matrix contains values different from 0, 1 or NaN."
            )

    def __len__(self):
        return self.maldi.shape[0]

    def __getitem__(self, idx):
        maldi = self.maldi[idx]
        amr_vec = self.amr[idx].copy()

        mask = ~np.isnan(amr_vec)

        labels = amr_vec.copy()
        labels[~mask] = 0

        return (
            torch.tensor(maldi).float(),
            torch.tensor(labels).float(),
            torch.tensor(mask).float()
        )


# ============================================================
# MODEL
# ============================================================
class MaldiEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        emb_dim
    ):
        super().__init__()

        dims = [input_dim] + list(hidden_dims)

        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.2))

        # Última proyección al embedding final
        if dims[-1] != emb_dim:
            layers.append(nn.Linear(dims[-1], emb_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim
    ):
        super().__init__()

        sizes = [input_dim] + list(hidden_dims) + [output_dim]

        layers = []

        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TwoStageContextNCF(pl.LightningModule):
    """
    Arquitectura nueva:

    Stage 1:
        MALDI embedding + antibiotic embedding -> logit por antibiótico

    Se concatenan las predicciones individuales en un vector:
        logits_stage1/probs_stage1 = pseudo-antibiograma

    Stage 2:
        MALDI embedding + antibiotic embedding + pseudo-contexto -> logit final

    Loss:
        loss = loss_stage2 + lambda_stage1 * loss_stage1

    Context modes:
        - threshold: usa solo predicciones muy confiadas.
        - topk: usa los k antibióticos más confiados.
        - oracle: usa resistencias reales parciales como techo.
    """

    def __init__(
        self,
        num_feat,
        num_items,
        context_mode,
        low_threshold=None,
        high_threshold=None,
        top_k=None,
        oracle_fraction=None,
        oracle_noise=0.0,
        drug_emb_dim=32,
        maldi_hidden_dims=None,
        maldi_emb_dim=128,
        stage1_hidden_dims=None,
        stage2_hidden_dims=None,
        lambda_stage1=0.3,
        lr=1e-3
    ):
        super().__init__()

        self.save_hyperparameters()

        self.num_items = num_items
        self.context_mode = context_mode
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.top_k = top_k
        self.oracle_fraction = oracle_fraction
        self.oracle_noise = oracle_noise

        self.lambda_stage1 = lambda_stage1
        self.lr = lr

        if maldi_hidden_dims is None:
            maldi_hidden_dims = [2048, 1024, 512, 256, 128]

        if stage1_hidden_dims is None:
            stage1_hidden_dims = [256, 128, 64]

        if stage2_hidden_dims is None:
            stage2_hidden_dims = [256, 128, 64]

        self.maldi_encoder = MaldiEncoder(
            input_dim=num_feat,
            hidden_dims=maldi_hidden_dims,
            emb_dim=maldi_emb_dim
        )

        self.drug_embedding = nn.Embedding(
            num_items,
            drug_emb_dim
        )

        # Stage 1:
        # predicción individual por antibiótico.
        stage1_input_dim = maldi_emb_dim + drug_emb_dim

        self.stage1_mlp = MLP(
            input_dim=stage1_input_dim,
            hidden_dims=stage1_hidden_dims,
            output_dim=1
        )

        # Stage 2:
        # vuelve a ver MALDI embedding + antibiotic embedding
        # + vector de contexto + máscara de contexto.
        stage2_input_dim = (
            maldi_emb_dim +
            drug_emb_dim +
            num_items +      # context_values
            num_items        # context_mask
        )

        self.stage2_mlp = MLP(
            input_dim=stage2_input_dim,
            hidden_dims=stage2_hidden_dims,
            output_dim=1
        )

        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    def move_batch(self, batch):
        device = self.device
        return [x.to(device) for x in batch]

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

    def compute_stage1_logits_from_emb(self, maldi_emb):
        """
        Calcula logits stage1 para todos los antibióticos.

        maldi_emb: [B, maldi_emb_dim]

        Output:
            logits_stage1: [B, num_items]
        """

        batch_size = maldi_emb.shape[0]
        device = maldi_emb.device

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

    # ------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------
    def build_threshold_context(self, probs_stage1):
        """
        Contexto selectivo por confianza.

        p >= high_threshold -> visible como 1
        p <= low_threshold  -> visible como 0
        resto               -> missing

        Devuelve:
            context_values_base: [B, N]
            context_mask_base:   [B, N]
        """

        high = self.high_threshold
        low = self.low_threshold

        resistant = probs_stage1 >= high
        susceptible = probs_stage1 <= low

        confident = resistant | susceptible

        context_values = resistant.float()
        context_mask = confident.float()

        return context_values, context_mask

    def build_topk_context(self, probs_stage1):
        """
        Selecciona los top-k antibióticos más confiados por muestra.

        confidence = |p - 0.5|

        Para los seleccionados:
            value = 1 si p >= 0.5
            value = 0 si p < 0.5
        """

        batch_size, num_items = probs_stage1.shape

        k = self.top_k

        if k is None or k <= 0:
            context_values = torch.zeros_like(probs_stage1)
            context_mask = torch.zeros_like(probs_stage1)
            return context_values, context_mask

        k = min(k, num_items)

        confidence = torch.abs(probs_stage1 - 0.5)

        _, idx = torch.topk(
            confidence,
            k=k,
            dim=1
        )

        context_mask = torch.zeros_like(probs_stage1)
        context_mask.scatter_(1, idx, 1.0)

        context_values = (probs_stage1 >= 0.5).float()
        context_values = context_values * context_mask

        return context_values, context_mask

    def build_oracle_context(self, labels, mask):
        """
        Techo del pseudo-contexto.

        Usa una fracción de resistencias reales observadas como contexto.

        Importante:
            todavía se excluirá el target antibiótico después.
        """

        batch_size, num_items = labels.shape
        device = labels.device

        if self.oracle_fraction is None:
            fraction = 0.5
        else:
            fraction = self.oracle_fraction

        random_keep = torch.rand(
            batch_size,
            num_items,
            device=device
        ) < fraction

        context_mask = mask.float() * random_keep.float()

        context_values = labels.float() * context_mask

        if self.oracle_noise > 0:
            noise_flip = (
                torch.rand(batch_size, num_items, device=device)
                < self.oracle_noise
            ).float()

            flipped_values = 1.0 - context_values

            context_values = torch.where(
                (context_mask > 0) & (noise_flip > 0),
                flipped_values,
                context_values
            )

        return context_values, context_mask

    def build_context_base(self, logits_stage1, labels=None, mask=None):
        """
        Construye el contexto base [B, N] antes de excluir el target.

        Según el experimento:
            threshold
            topk
            oracle
        """

        probs_stage1 = torch.sigmoid(logits_stage1)

        if self.context_mode == "threshold":
            return self.build_threshold_context(probs_stage1)

        if self.context_mode == "topk":
            return self.build_topk_context(probs_stage1)

        if self.context_mode == "oracle":
            if labels is None or mask is None:
                raise ValueError(
                    "Oracle context requires labels and mask."
                )
            return self.build_oracle_context(labels, mask)

        raise ValueError(
            f"Unknown context_mode: {self.context_mode}"
        )

    def expand_context_per_target(
        self,
        context_values_base,
        context_mask_base
    ):
        """
        Convierte contexto [B, N] en contexto target-specific [B, N, N].

        Para cada target j, se elimina la posición j del contexto.

        Output:
            context_values_per_target: [B, N_targets, N_context]
            context_mask_per_target:   [B, N_targets, N_context]
        """

        batch_size, num_items = context_values_base.shape
        device = context_values_base.device

        values = context_values_base.unsqueeze(1).expand(
            batch_size,
            num_items,
            num_items
        ).clone()

        masks = context_mask_base.unsqueeze(1).expand(
            batch_size,
            num_items,
            num_items
        ).clone()

        eye = torch.eye(
            num_items,
            device=device
        ).unsqueeze(0)

        target_exclusion = 1.0 - eye

        values = values * target_exclusion
        masks = masks * target_exclusion

        return values, masks

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, maldi, labels=None, mask=None):
        """
        Devuelve:
            logits_stage2: predicción final [B, N]
            logits_stage1: predicción primera MLP [B, N]
            context_values_base: [B, N]
            context_mask_base:   [B, N]
        """

        batch_size = maldi.shape[0]
        device = maldi.device

        maldi_emb = self.maldi_encoder(maldi)

        # Stage 1: pseudo-antibiograma
        logits_stage1 = self.compute_stage1_logits_from_emb(maldi_emb)

        # Construir contexto base desde stage1 o desde oracle
        context_values_base, context_mask_base = self.build_context_base(
            logits_stage1=logits_stage1,
            labels=labels,
            mask=mask
        )

        # Contexto específico para cada target:
        # para target j, eliminamos context[j].
        context_values_per_target, context_mask_per_target = (
            self.expand_context_per_target(
                context_values_base,
                context_mask_base
            )
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

        # Stage 2 ve:
        # MALDI embedding + antibiotic embedding + context_values + context_mask
        stage2_input = torch.cat(
            [
                maldi_expanded,
                drug_expanded,
                context_values_per_target,
                context_mask_per_target,
            ],
            dim=-1
        )

        stage2_input = stage2_input.reshape(
            batch_size * self.num_items,
            -1
        )

        logits_stage2 = self.stage2_mlp(stage2_input)
        logits_stage2 = logits_stage2.view(
            batch_size,
            self.num_items
        )

        return (
            logits_stage2,
            logits_stage1,
            context_values_base,
            context_mask_base
        )

    # ------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        maldi, labels, mask = self.move_batch(batch)

        logits_stage2, logits_stage1, _, _ = self.forward(
            maldi=maldi,
            labels=labels,
            mask=mask
        )

        loss_stage1 = self.masked_bce(
            logits=logits_stage1,
            labels=labels,
            mask=mask
        )

        loss_stage2 = self.masked_bce(
            logits=logits_stage2,
            labels=labels,
            mask=mask
        )

        loss = loss_stage2 + self.lambda_stage1 * loss_stage1

        self.log("loss_tr", loss, prog_bar=True)
        self.log("loss_stage1_tr", loss_stage1, prog_bar=False)
        self.log("loss_stage2_tr", loss_stage2, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        maldi, labels, mask = self.move_batch(batch)

        logits_stage2, logits_stage1, _, _ = self.forward(
            maldi=maldi,
            labels=labels,
            mask=mask
        )

        loss_stage1 = self.masked_bce(
            logits=logits_stage1,
            labels=labels,
            mask=mask
        )

        loss_stage2 = self.masked_bce(
            logits=logits_stage2,
            labels=labels,
            mask=mask
        )

        loss = loss_stage2 + self.lambda_stage1 * loss_stage1

        self.log("loss_val", loss, prog_bar=True)
        self.log("loss_stage1_val", loss_stage1, prog_bar=False)
        self.log("loss_stage2_val", loss_stage2, prog_bar=False)

        return loss

    def predict_proba(self, maldi, labels=None, mask=None):
        self.eval()

        with torch.no_grad():
            logits_stage2, logits_stage1, context_values, context_mask = (
                self.forward(
                    maldi=maldi,
                    labels=labels,
                    mask=mask
                )
            )

            probs_stage2 = torch.sigmoid(logits_stage2)
            probs_stage1 = torch.sigmoid(logits_stage1)

        return probs_stage2, probs_stage1, context_values, context_mask

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ============================================================
# DATA HELPERS
# ============================================================
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


def create_folds(n_samples, n_splits=5):

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    return list(kf.split(np.arange(n_samples)))


def split_train_val_indices(train_idx, val_size=0.2, seed=42):

    train_inner_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_size,
        random_state=seed,
        shuffle=True
    )

    return train_inner_idx, val_idx


def make_loader(X, amr, batch_size, shuffle, num_workers):

    dataset = AMRVectorDataset(
        maldi=X,
        amr=amr
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader


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
        auc_macro = np.mean(antibiotic_aucs)

    return auc_micro, auc_macro


def compute_context_coverage(y_true, context_mask):

    valid = ~np.isnan(y_true)

    if np.sum(valid) == 0:
        return np.nan

    visible = (context_mask > 0) & valid

    return float(np.sum(visible) / np.sum(valid))


# ============================================================
# PREDICTION
# ============================================================
def predict_model(model, X, amr=None, batch_size=512):
    """
    Para modos threshold/topk:
        amr puede ser None.

    Para modo oracle:
        necesitamos amr para construir contexto real parcial.
    """

    model.eval()

    stage2_preds = []
    stage1_preds = []
    context_masks = []

    with torch.no_grad():

        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])

            x_batch = torch.tensor(
                X[start:end]
            ).float().to(model.device)

            if amr is not None:
                amr_batch_np = amr[start:end].copy()
                mask_np = ~np.isnan(amr_batch_np)

                labels_np = amr_batch_np.copy()
                labels_np[~mask_np] = 0

                labels_batch = torch.tensor(
                    labels_np
                ).float().to(model.device)

                mask_batch = torch.tensor(
                    mask_np
                ).float().to(model.device)

            else:
                labels_batch = None
                mask_batch = None

            probs_stage2, probs_stage1, _, context_mask = (
                model.predict_proba(
                    maldi=x_batch,
                    labels=labels_batch,
                    mask=mask_batch
                )
            )

            stage2_preds.append(probs_stage2.cpu().numpy())
            stage1_preds.append(probs_stage1.cpu().numpy())
            context_masks.append(context_mask.cpu().numpy())

    stage2_preds = np.vstack(stage2_preds)
    stage1_preds = np.vstack(stage1_preds)
    context_masks = np.vstack(context_masks)

    return stage2_preds, stage1_preds, context_masks


# ============================================================
# TRAIN ONE MODEL
# ============================================================
def train_one_experiment(
    experiment,
    species,
    fold,
    X_tr,
    amr_tr,
    X_val,
    amr_val,
    X_tst,
    amr_tst,
    num_items
):

    exp_name = experiment["name"]

    print(
        f"\nTraining experiment={exp_name} | "
        f"species={species} | fold={fold}",
        flush=True
    )

    loader_tr = make_loader(
        X=X_tr,
        amr=amr_tr,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    loader_val = make_loader(
        X=X_val,
        amr=amr_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    model = TwoStageContextNCF(
        num_feat=X_tr.shape[1],
        num_items=num_items,
        context_mode=experiment["context_mode"],
        low_threshold=experiment["low_threshold"],
        high_threshold=experiment["high_threshold"],
        top_k=experiment["top_k"],
        oracle_fraction=experiment["oracle_fraction"],
        oracle_noise=experiment["oracle_noise"],
        drug_emb_dim=DRUG_EMB_DIM,
        maldi_hidden_dims=MALDI_HIDDEN_DIMS,
        maldi_emb_dim=MALDI_EMB_DIM,
        stage1_hidden_dims=STAGE1_HIDDEN_DIMS,
        stage2_hidden_dims=STAGE2_HIDDEN_DIMS,
        lambda_stage1=LAMBDA_STAGE1,
        lr=LR
    )

    callbacks = [
        EarlyStopping(
            monitor="loss_val",
            patience=PATIENCE,
            mode="min"
        )
    ]

    checkpoint_callback = None

    safe_species = str(species).replace("/", "_").replace(" ", "_")

    if USE_CHECKPOINT:
        checkpoint_callback = ModelCheckpoint(
            monitor="loss_val",
            mode="min",
            save_top_k=1,
            filename=(
                f"{safe_species}_{exp_name}_fold{fold}"
                + "-{epoch:02d}-{loss_val:.4f}"
            )
        )

        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
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
            print(f"Loading best checkpoint: {best_path}", flush=True)

            model = TwoStageContextNCF.load_from_checkpoint(best_path)
            model = model.to(DEVICE)

    # Para oracle necesitamos pasar amr_tst para construir contexto real parcial.
    if experiment["context_mode"] == "oracle":
        amr_for_prediction = amr_tst
    else:
        amr_for_prediction = None

    stage2_preds, stage1_preds, context_masks = predict_model(
        model=model,
        X=X_tst,
        amr=amr_for_prediction
    )

    stage2_micro, stage2_macro = compute_metrics(
        y_true=amr_tst,
        preds=stage2_preds
    )

    stage1_micro, stage1_macro = compute_metrics(
        y_true=amr_tst,
        preds=stage1_preds
    )

    context_coverage = compute_context_coverage(
        y_true=amr_tst,
        context_mask=context_masks
    )

    print(
        f"Result {exp_name} | "
        f"stage2_micro={stage2_micro:.4f} | "
        f"stage2_macro={stage2_macro:.4f} | "
        f"stage1_micro={stage1_micro:.4f} | "
        f"stage1_macro={stage1_macro:.4f} | "
        f"context_coverage={context_coverage:.4f}",
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
        "oracle_fraction": experiment["oracle_fraction"],
        "oracle_noise": experiment["oracle_noise"],
        "stage2_auc_micro": stage2_micro,
        "stage2_auc_macro": stage2_macro,
        "stage1_auc_micro": stage1_micro,
        "stage1_auc_macro": stage1_macro,
        "context_coverage": context_coverage,
        "lambda_stage1": LAMBDA_STAGE1,
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
        print("Skipping...", flush=True)
        return []

    X, amr, antibiotic_names = data

    num_items = amr.shape[1]
    n_samples = X.shape[0]

    print(
        f"Samples: {n_samples} | Items: {num_items}",
        flush=True
    )

    folds = create_folds(
        n_samples=n_samples,
        n_splits=N_SPLITS
    )

    species_results = []

    for fold, (train_idx, test_idx) in enumerate(folds):

        print(f"\nFold {fold}", flush=True)

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

        for experiment in EXPERIMENTS:

            try:
                res = train_one_experiment(
                    experiment=experiment,
                    species=species,
                    fold=fold,
                    X_tr=X_tr,
                    amr_tr=amr_tr,
                    X_val=X_val,
                    amr_val=amr_val,
                    X_tst=X_tst,
                    amr_tst=amr_tst,
                    num_items=num_items
                )

                species_results.append(res)

                # Guardado incremental por seguridad
                partial_df = pd.DataFrame(species_results)

                partial_path = os.path.join(
                    OUTPUT_DIR,
                    f"{str(species).replace('/', '_').replace(' ', '_')}_partial_results.csv"
                )

                partial_df.to_csv(partial_path, index=False)

            except Exception as e:
                print(
                    f"Error | species={species} | "
                    f"experiment={experiment['name']} | fold={fold}: {e}",
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

        df_all_partial = pd.DataFrame(all_results)

        partial_global_path = os.path.join(
            OUTPUT_DIR,
            "all_two_stage_context_partial_results.csv"
        )

        df_all_partial.to_csv(partial_global_path, index=False)

    except Exception as e:
        print(f"Error in species {sp}: {e}", flush=True)


df_all = pd.DataFrame(all_results)

raw_path = os.path.join(
    OUTPUT_DIR,
    "two_stage_context_raw_fold_results.csv"
)

df_all.to_csv(raw_path, index=False)

print("\nRAW FOLD RESULTS:")
print(df_all)
print(f"\nSaved raw fold results: {raw_path}")


# ============================================================
# SUMMARY BY SPECIES AND EXPERIMENT
# ============================================================
if len(df_all) > 0:

    summary = (
        df_all
        .groupby(["experiment", "species"], as_index=False)
        .agg(
            stage2_auc_micro=("stage2_auc_micro", "mean"),
            stage2_auc_macro=("stage2_auc_macro", "mean"),
            stage2_auc_micro_std=("stage2_auc_micro", "std"),
            stage2_auc_macro_std=("stage2_auc_macro", "std"),
            stage1_auc_micro=("stage1_auc_micro", "mean"),
            stage1_auc_macro=("stage1_auc_macro", "mean"),
            context_coverage=("context_coverage", "mean"),
            lambda_stage1=("lambda_stage1", "mean"),
        )
    )

    summary = summary.sort_values(
        by=["experiment", "stage2_auc_macro"],
        ascending=[True, False]
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "two_stage_context_summary_by_species.csv"
    )

    summary.to_csv(summary_path, index=False)

    print("\nSUMMARY BY SPECIES:")
    print(summary)
    print(f"\nSaved summary by species: {summary_path}")


    # ========================================================
    # GLOBAL SUMMARY BY EXPERIMENT
    # ========================================================
    global_summary = (
        summary
        .groupby("experiment", as_index=False)
        .agg(
            mean_stage2_auc_micro=("stage2_auc_micro", "mean"),
            mean_stage2_auc_macro=("stage2_auc_macro", "mean"),
            mean_stage1_auc_micro=("stage1_auc_micro", "mean"),
            mean_stage1_auc_macro=("stage1_auc_macro", "mean"),
            mean_context_coverage=("context_coverage", "mean"),
        )
    )

    global_summary = global_summary.sort_values(
        by="mean_stage2_auc_macro",
        ascending=False
    )

    global_summary_path = os.path.join(
        OUTPUT_DIR,
        "two_stage_context_global_summary.csv"
    )

    global_summary.to_csv(global_summary_path, index=False)

    print("\nGLOBAL SUMMARY BY EXPERIMENT:")
    print(global_summary)
    print(f"\nSaved global summary: {global_summary_path}")


print("\nDone.")