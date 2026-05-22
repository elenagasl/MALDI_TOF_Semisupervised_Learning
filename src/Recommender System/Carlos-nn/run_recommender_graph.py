import os
import pickle
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from pytorch_lightning.callbacks import EarlyStopping

from lib.GraphRecDataset import GraphRecDataset
from lib.GraphNCF import GraphNCF


# =========================
# CONFIG
# =========================
DATA_PATH = '/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl'

N_SPLITS = 2
VAL_SIZE = 0.2

BATCH_SIZE = 64
MAX_EPOCHS = 300
PATIENCE = 15
LR = 1e-3

# Graph config
INIT_ALPHA = 0.2
TRAIN_ALPHA = True
LAMBDA_GRAPH_REG = 0.01
MIN_CORR_PAIR = 30

# W construction
USE_ABS_CORRELATION = False
KEEP_POSITIVE_ONLY = True

# Output
GRAPH_DIR = "learned_graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", DEVICE, flush=True)

torch.backends.cudnn.benchmark = True


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


# =========================
# BUILD SPECIES DATA
# =========================
def build_species_data(species):

    mask = (y_species_all == species)

    X = X_all[mask]
    amr = amr_all[mask].copy()

    # =========================
    # CLEAN AMR VALUES
    # =========================
    # Only binary labels:
    # 0 = susceptible
    # 1 = resistant
    # everything else becomes NaN
    valid_values = (amr == 0) | (amr == 1) | np.isnan(amr)
    amr[~valid_values] = np.nan

    if len(X) < 100:
        return None

    # =========================
    # REMOVE LOW-INFORMATION ANTIBIOTICS
    # =========================
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


# =========================
# FOLDS
# =========================
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


# =========================
# PHI CORRELATION MATRIX
# =========================
def compute_phi_correlation_matrix(
    amr_train,
    min_pair=30,
    use_abs=False,
    positive_only=True
):
    """
    Computes phi correlation between antibiotics using only train data.

    Output:
        W_init: shape [num_items, num_items]

    Notes:
        - NaNs are ignored pairwise.
        - Diagonal is set to 0.
        - If positive_only=True, negative correlations are removed.
        - Rows are normalized later in GraphNCF.
    """

    num_items = amr_train.shape[1]
    W = np.zeros((num_items, num_items), dtype=np.float32)

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

            phi = ((n11 * n00) - (n10 * n01)) / denom

            if use_abs:
                phi = abs(phi)

            if positive_only and phi < 0:
                phi = 0.0

            W[i, j] = phi
            W[j, i] = phi

    np.fill_diagonal(W, 0.0)

    W = np.nan_to_num(
        W,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    return W


# =========================
# LOADERS
# =========================
def make_loader(X, amr, batch_size, shuffle, num_workers):

    dataset = GraphRecDataset(
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


# =========================
# PREDICT
# =========================
def predict_all_antibiotics(model, X, batch_size=512):

    model.eval()

    preds = []

    with torch.no_grad():

        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])

            x_batch = torch.tensor(X[start:end]).float().to(model.device)

            logits = model(x_batch)
            probas = torch.sigmoid(logits)

            preds.append(probas.cpu().numpy())

    preds = np.vstack(preds)

    return preds


# =========================
# METRICS
# =========================
def compute_metrics(y_true, preds):

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
        auc_macro = np.mean(antibiotic_aucs)

    return auc_micro, auc_macro


# =========================
# TRAIN MODEL
# =========================
def train_graph_model(
    model_name,
    X_tr,
    amr_tr,
    X_val,
    amr_val,
    X_tst,
    amr_tst,
    num_items,
    W_init,
    graph_mode,
    n_graph_steps,
    species,
    fold
):

    print(
        f"\nTraining {model_name} | "
        f"graph_mode={graph_mode} | "
        f"steps={n_graph_steps}",
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

    if graph_mode == "none":
        W_for_model = None
        steps_for_model = 0
    else:
        W_for_model = W_init
        steps_for_model = n_graph_steps

    model = GraphNCF(
        num_feat=X_tr.shape[1],
        num_items=num_items,
        graph_mode=graph_mode,
        W_init=W_for_model,
        n_graph_steps=steps_for_model,
        init_alpha=INIT_ALPHA,
        train_alpha=TRAIN_ALPHA,
        lr=LR,
        lambda_graph_reg=LAMBDA_GRAPH_REG
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if DEVICE == "cuda" else "cpu",
        devices=1,
        callbacks=[
            EarlyStopping(
                monitor="loss_val",
                patience=PATIENCE,
                mode="min"
            )
        ],
        logger=False,
        enable_checkpointing=False
    )

    trainer.fit(model, loader_tr, loader_val)

    preds = predict_all_antibiotics(
        model=model,
        X=X_tst
    )

    auc_micro, auc_macro = compute_metrics(
        y_true=amr_tst,
        preds=preds
    )

    alpha_value = np.nan

    if graph_mode != "none":
        alpha_value = float(
            model.graph_layer.get_alpha().detach().cpu().numpy()
        )

    print(
        f"{model_name} result | "
        f"auc_micro={auc_micro:.4f} | "
        f"auc_macro={auc_macro:.4f} | "
        f"alpha={alpha_value}",
        flush=True
    )

    # =========================
    # SAVE LEARNED W
    # =========================
    if graph_mode == "trainable":

        W_learned = model.graph_layer.get_W().detach().cpu().numpy()

        safe_species = str(species).replace("/", "_").replace(" ", "_")

        W_path = os.path.join(
            GRAPH_DIR,
            f"{safe_species}_fold{fold}_{model_name}_W.csv"
        )

        pd.DataFrame(W_learned).to_csv(
            W_path,
            index=False
        )

        print(f"Saved learned W: {W_path}", flush=True)

    return auc_micro, auc_macro, alpha_value


# =========================
# TRAIN ONE SPECIES
# =========================
def train_species(species):

    print("\n====================", flush=True)
    print(f"Species: {species}", flush=True)
    print("====================", flush=True)

    data = build_species_data(species)

    if data is None:
        print("Skipping...", flush=True)
        return None, None, None, None

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

    base_micro = []
    base_macro = []
    base_alpha = []

    fixed_micro = []
    fixed_macro = []
    fixed_alpha = []

    trainable_micro = []
    trainable_macro = []
    trainable_alpha = []

    trainable3_micro = []
    trainable3_macro = []
    trainable3_alpha = []

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

        # =========================
        # W INIT FROM TRAIN ONLY
        # =========================
        W_init = compute_phi_correlation_matrix(
            amr_train=amr_tr,
            min_pair=MIN_CORR_PAIR,
            use_abs=USE_ABS_CORRELATION,
            positive_only=KEEP_POSITIVE_ONLY
        )

        safe_species = str(species).replace("/", "_").replace(" ", "_")

        W_init_path = os.path.join(
            GRAPH_DIR,
            f"{safe_species}_fold{fold}_W_init.csv"
        )

        pd.DataFrame(
            W_init,
            index=antibiotic_names,
            columns=antibiotic_names
        ).to_csv(W_init_path)

        print(
            f"W_init stats | "
            f"nonzero={np.sum(W_init > 0)} | "
            f"mean={np.mean(W_init):.4f} | "
            f"max={np.max(W_init):.4f}",
            flush=True
        )

        # =========================
        # 1) BASE MODEL
        # =========================
        micro, macro, alpha = train_graph_model(
            model_name="graph_base",
            X_tr=X_tr,
            amr_tr=amr_tr,
            X_val=X_val,
            amr_val=amr_val,
            X_tst=X_tst,
            amr_tst=amr_tst,
            num_items=num_items,
            W_init=W_init,
            graph_mode="none",
            n_graph_steps=0,
            species=species,
            fold=fold
        )

        if not np.isnan(micro):
            base_micro.append(micro)

        if not np.isnan(macro):
            base_macro.append(macro)

        base_alpha.append(alpha)

        # =========================
        # 2) FIXED GRAPH, 1 STEP
        # =========================
        micro, macro, alpha = train_graph_model(
            model_name="graph_fixed",
            X_tr=X_tr,
            amr_tr=amr_tr,
            X_val=X_val,
            amr_val=amr_val,
            X_tst=X_tst,
            amr_tst=amr_tst,
            num_items=num_items,
            W_init=W_init,
            graph_mode="fixed",
            n_graph_steps=1,
            species=species,
            fold=fold
        )

        if not np.isnan(micro):
            fixed_micro.append(micro)

        if not np.isnan(macro):
            fixed_macro.append(macro)

        fixed_alpha.append(alpha)

        # =========================
        # 3) TRAINABLE GRAPH, 1 STEP
        # =========================
        micro, macro, alpha = train_graph_model(
            model_name="graph_trainable",
            X_tr=X_tr,
            amr_tr=amr_tr,
            X_val=X_val,
            amr_val=amr_val,
            X_tst=X_tst,
            amr_tst=amr_tst,
            num_items=num_items,
            W_init=W_init,
            graph_mode="trainable",
            n_graph_steps=1,
            species=species,
            fold=fold
        )

        if not np.isnan(micro):
            trainable_micro.append(micro)

        if not np.isnan(macro):
            trainable_macro.append(macro)

        trainable_alpha.append(alpha)

        # =========================
        # 4) TRAINABLE GRAPH, 3 STEPS
        # =========================
        micro, macro, alpha = train_graph_model(
            model_name="graph_trainable_3_steps",
            X_tr=X_tr,
            amr_tr=amr_tr,
            X_val=X_val,
            amr_val=amr_val,
            X_tst=X_tst,
            amr_tst=amr_tst,
            num_items=num_items,
            W_init=W_init,
            graph_mode="trainable",
            n_graph_steps=3,
            species=species,
            fold=fold
        )

        if not np.isnan(micro):
            trainable3_micro.append(micro)

        if not np.isnan(macro):
            trainable3_macro.append(macro)

        trainable3_alpha.append(alpha)

    base_result = {
        "species": species,
        "auc_micro": np.mean(base_micro) if base_micro else np.nan,
        "auc_macro": np.mean(base_macro) if base_macro else np.nan,
        "auc_micro_std": np.std(base_micro) if base_micro else np.nan,
        "auc_macro_std": np.std(base_macro) if base_macro else np.nan,
        "alpha_mean": np.nanmean(base_alpha) if base_alpha else np.nan
    }

    fixed_result = {
        "species": species,
        "auc_micro": np.mean(fixed_micro) if fixed_micro else np.nan,
        "auc_macro": np.mean(fixed_macro) if fixed_macro else np.nan,
        "auc_micro_std": np.std(fixed_micro) if fixed_micro else np.nan,
        "auc_macro_std": np.std(fixed_macro) if fixed_macro else np.nan,
        "alpha_mean": np.nanmean(fixed_alpha) if fixed_alpha else np.nan
    }

    trainable_result = {
        "species": species,
        "auc_micro": np.mean(trainable_micro) if trainable_micro else np.nan,
        "auc_macro": np.mean(trainable_macro) if trainable_macro else np.nan,
        "auc_micro_std": np.std(trainable_micro) if trainable_micro else np.nan,
        "auc_macro_std": np.std(trainable_macro) if trainable_macro else np.nan,
        "alpha_mean": np.nanmean(trainable_alpha) if trainable_alpha else np.nan
    }

    trainable3_result = {
        "species": species,
        "auc_micro": np.mean(trainable3_micro) if trainable3_micro else np.nan,
        "auc_macro": np.mean(trainable3_macro) if trainable3_macro else np.nan,
        "auc_micro_std": np.std(trainable3_micro) if trainable3_micro else np.nan,
        "auc_macro_std": np.std(trainable3_macro) if trainable3_macro else np.nan,
        "alpha_mean": np.nanmean(trainable3_alpha) if trainable3_alpha else np.nan
    }

    return base_result, fixed_result, trainable_result, trainable3_result


# =========================
# MAIN
# =========================
base_results = []
fixed_results = []
trainable_results = []
trainable3_results = []

for sp in species_list:

    try:
        base_res, fixed_res, trainable_res, trainable3_res = train_species(sp)

        if base_res is not None:
            base_results.append(base_res)

        if fixed_res is not None:
            fixed_results.append(fixed_res)

        if trainable_res is not None:
            trainable_results.append(trainable_res)

        if trainable3_res is not None:
            trainable3_results.append(trainable3_res)

    except Exception as e:
        print(f"Error in {sp}: {e}", flush=True)


df_base = pd.DataFrame(base_results)
df_fixed = pd.DataFrame(fixed_results)
df_trainable = pd.DataFrame(trainable_results)
df_trainable3 = pd.DataFrame(trainable3_results)

if len(df_base) > 0:
    df_base = df_base.sort_values(by="auc_macro", ascending=False)

if len(df_fixed) > 0:
    df_fixed = df_fixed.sort_values(by="auc_macro", ascending=False)

if len(df_trainable) > 0:
    df_trainable = df_trainable.sort_values(by="auc_macro", ascending=False)

if len(df_trainable3) > 0:
    df_trainable3 = df_trainable3.sort_values(by="auc_macro", ascending=False)

print("\nFINAL GRAPH BASE RESULTS:")
print(df_base)

print("\nFINAL GRAPH FIXED RESULTS:")
print(df_fixed)

print("\nFINAL GRAPH TRAINABLE RESULTS:")
print(df_trainable)

print("\nFINAL GRAPH TRAINABLE 3 STEPS RESULTS:")
print(df_trainable3)

df_base.to_csv("graph_base.csv", index=False)
df_fixed.to_csv("graph_fixed.csv", index=False)
df_trainable.to_csv("graph_trainable.csv", index=False)
df_trainable3.to_csv("graph_trainable_3_steps.csv", index=False)

print("\nSaved:")
print("graph_base.csv")
print("graph_fixed.csv")
print("graph_trainable.csv")
print("graph_trainable_3_steps.csv")
print(f"Learned graphs saved in: {GRAPH_DIR}/")