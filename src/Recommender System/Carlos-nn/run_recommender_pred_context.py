import os
import pickle
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from lib.PredContextRecDataset import PredContextRecDataset
from lib.PredContextNCF import PredContextNCF


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

DRUG_EMB_DIM = 32
HIDDEN_DIMS = [128, 64]

# Peso de la loss auxiliar:
# MALDI -> vector global AMR
LAMBDA_CONTEXT = 0.3

# Probabilidad de usar contexto real enmascarado durante training.
# El resto de veces se usa contexto predicho.
REAL_CONTEXT_PROB = 0.5

# Dropout sobre el contexto real cuando se usa.
# Simula que solo vemos parte del antibiograma.
CONTEXT_DROPOUT = 0.3

# Si True, no deja que la loss target actualice la context head.
# Yo empezaría con False.
DETACH_PRED_CONTEXT = False

USE_CHECKPOINT = True

# Carpeta donde se guardan los análisis de la cabeza contextual
CONTEXT_ANALYSIS_DIR = "context_head_analysis"
os.makedirs(CONTEXT_ANALYSIS_DIR, exist_ok=True)

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

    # Solo labels binarios:
    # 0 = susceptible
    # 1 = resistant
    # todo lo demás se convierte en NaN
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
# LOADERS
# =========================
def make_loader(X, amr, batch_size, shuffle, num_workers):

    dataset = PredContextRecDataset(
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
# PREDICT ALL ANTIBIOTICS
# =========================
def predict_all_antibiotics(model, X, batch_size=512):

    model.eval()

    preds = []

    with torch.no_grad():

        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])

            x_batch = torch.tensor(X[start:end]).float().to(model.device)

            probas = model.predict_all_antibiotics(x_batch)

            preds.append(probas.cpu().numpy())

    preds = np.vstack(preds)

    return preds


# =========================
# PREDICT PSEUDO-CONTEXT HEAD
# =========================
def predict_context_head(model, X, batch_size=512):
    """
    Evalúa únicamente la cabeza auxiliar:
        MALDI -> predicted AMR vector

    Devuelve:
        context_preds: matriz [n_samples, num_items]
        con probabilidades de resistencia por antibiótico.
    """

    model.eval()

    preds = []

    with torch.no_grad():

        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])

            x_batch = torch.tensor(X[start:end]).float().to(model.device)

            maldi_emb = model.maldi_encoder(x_batch)
            context_logits = model.context_head(maldi_emb)
            context_probs = torch.sigmoid(context_logits)

            preds.append(context_probs.cpu().numpy())

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
# CONTEXT HEAD METRICS BY ANTIBIOTIC
# =========================
def compute_antibiotic_level_metrics(y_true, preds, antibiotic_names=None):
    """
    Calcula AUC por antibiótico para la context_head.

    y_true:
        matriz [n_samples, num_items] con 0/1/NaN.

    preds:
        matriz [n_samples, num_items] con probabilidades predichas.

    antibiotic_names:
        nombres de antibióticos filtrados para esa especie.
    """

    rows = []

    num_items = y_true.shape[1]

    for j in range(num_items):

        col = y_true[:, j]
        valid = ~np.isnan(col)

        n_obs = int(np.sum(valid))

        if n_obs == 0:
            auc = np.nan
            n_pos = 0
            n_neg = 0

        else:
            n_pos = int(np.sum(col[valid] == 1))
            n_neg = int(np.sum(col[valid] == 0))

            if len(np.unique(col[valid])) > 1:
                auc = roc_auc_score(
                    col[valid],
                    preds[valid, j]
                )
            else:
                auc = np.nan

        if antibiotic_names is not None:
            antibiotic_name = antibiotic_names[j]
        else:
            antibiotic_name = f"drug_{j}"

        rows.append({
            "antibiotic_idx": j,
            "antibiotic": antibiotic_name,
            "auc": auc,
            "n_obs": n_obs,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "resistance_rate": n_pos / n_obs if n_obs > 0 else np.nan
        })

    return pd.DataFrame(rows)


# =========================
# CONTEXT HEAD SUMMARY
# =========================
def evaluate_context_head(
    model,
    X,
    amr,
    species,
    fold,
    antibiotic_names=None,
    output_dir=CONTEXT_ANALYSIS_DIR
):
    """
    Evalúa la calidad del pseudo-contexto AMR predicho desde MALDI.

    Guarda:
        1) AUC micro/macro global de context_head.
        2) CSV con AUC por antibiótico.
        3) CSV con predicciones completas del pseudo-contexto.
    """

    os.makedirs(output_dir, exist_ok=True)

    context_preds = predict_context_head(
        model=model,
        X=X
    )

    context_micro, context_macro = compute_metrics(
        y_true=amr,
        preds=context_preds
    )

    df_abx = compute_antibiotic_level_metrics(
        y_true=amr,
        preds=context_preds,
        antibiotic_names=antibiotic_names
    )

    safe_species = str(species).replace("/", "_").replace(" ", "_")

    abx_path = os.path.join(
        output_dir,
        f"{safe_species}_fold{fold}_context_head_antibiotic_auc.csv"
    )

    preds_path = os.path.join(
        output_dir,
        f"{safe_species}_fold{fold}_context_head_predictions.csv"
    )

    df_abx.to_csv(abx_path, index=False)

    if antibiotic_names is not None:
        df_preds = pd.DataFrame(
            context_preds,
            columns=antibiotic_names
        )
    else:
        df_preds = pd.DataFrame(context_preds)

    df_preds.to_csv(preds_path, index=False)

    print(
        f"Context head result | "
        f"auc_micro={context_micro:.4f} | "
        f"auc_macro={context_macro:.4f}",
        flush=True
    )

    print(
        f"Saved context head antibiotic AUCs: {abx_path}",
        flush=True
    )

    return {
        "species": species,
        "fold": fold,
        "context_auc_micro": context_micro,
        "context_auc_macro": context_macro,
        "context_abx_auc_path": abx_path,
        "context_preds_path": preds_path
    }


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
        return None

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

    auc_micro_list = []
    auc_macro_list = []

    context_head_results = []

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

        model = PredContextNCF(
            num_feat=X_tr.shape[1],
            num_items=num_items,
            drug_emb_dim=DRUG_EMB_DIM,
            hidden_dim_CF=HIDDEN_DIMS,
            lr=LR,
            lambda_context=LAMBDA_CONTEXT,
            real_context_prob=REAL_CONTEXT_PROB,
            context_dropout=CONTEXT_DROPOUT,
            detach_pred_context=DETACH_PRED_CONTEXT
        )

        callbacks = [
            EarlyStopping(
                monitor="loss_val",
                patience=PATIENCE,
                mode="min"
            )
        ]

        checkpoint_callback = None

        if USE_CHECKPOINT:
            checkpoint_callback = ModelCheckpoint(
                monitor="loss_val",
                mode="min",
                save_top_k=1,
                filename=f"{species}_fold{fold}" + "-{epoch:02d}-{loss_val:.4f}"
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

                model = PredContextNCF.load_from_checkpoint(
                    best_path,
                    num_feat=X_tr.shape[1],
                    num_items=num_items,
                    drug_emb_dim=DRUG_EMB_DIM,
                    hidden_dim_CF=HIDDEN_DIMS,
                    lr=LR,
                    lambda_context=LAMBDA_CONTEXT,
                    real_context_prob=REAL_CONTEXT_PROB,
                    context_dropout=CONTEXT_DROPOUT,
                    detach_pred_context=DETACH_PRED_CONTEXT
                )

                model = model.to(DEVICE)

        # =========================
        # FINAL RECOMMENDER EVALUATION
        # =========================
        preds = predict_all_antibiotics(
            model=model,
            X=X_tst
        )

        auc_micro, auc_macro = compute_metrics(
            y_true=amr_tst,
            preds=preds
        )

        print(
            f"Fold {fold} final recommender result | "
            f"auc_micro={auc_micro:.4f} | "
            f"auc_macro={auc_macro:.4f}",
            flush=True
        )

        if not np.isnan(auc_micro):
            auc_micro_list.append(auc_micro)

        if not np.isnan(auc_macro):
            auc_macro_list.append(auc_macro)

        # =========================
        # AUXILIARY CONTEXT HEAD EVALUATION
        # =========================
        context_res = evaluate_context_head(
            model=model,
            X=X_tst,
            amr=amr_tst,
            species=species,
            fold=fold,
            antibiotic_names=antibiotic_names,
            output_dir=CONTEXT_ANALYSIS_DIR
        )

        context_head_results.append(context_res)

    context_micro_list = [
        r["context_auc_micro"]
        for r in context_head_results
        if not np.isnan(r["context_auc_micro"])
    ]

    context_macro_list = [
        r["context_auc_macro"]
        for r in context_head_results
        if not np.isnan(r["context_auc_macro"])
    ]

    result = {
        "species": species,

        # Final recommender performance
        "auc_micro": np.mean(auc_micro_list) if auc_micro_list else np.nan,
        "auc_macro": np.mean(auc_macro_list) if auc_macro_list else np.nan,
        "auc_micro_std": np.std(auc_micro_list) if auc_micro_list else np.nan,
        "auc_macro_std": np.std(auc_macro_list) if auc_macro_list else np.nan,

        # Auxiliary pseudo-context performance
        "context_auc_micro": np.mean(context_micro_list) if context_micro_list else np.nan,
        "context_auc_macro": np.mean(context_macro_list) if context_macro_list else np.nan,
        "context_auc_micro_std": np.std(context_micro_list) if context_micro_list else np.nan,
        "context_auc_macro_std": np.std(context_macro_list) if context_macro_list else np.nan,

        # Experiment parameters
        "lambda_context": LAMBDA_CONTEXT,
        "real_context_prob": REAL_CONTEXT_PROB,
        "context_dropout": CONTEXT_DROPOUT
    }

    return result


# =========================
# MAIN
# =========================
results = []

for sp in species_list:

    try:
        res = train_species(sp)

        if res is not None:
            results.append(res)

    except Exception as e:
        print(f"Error in {sp}: {e}", flush=True)


df = pd.DataFrame(results)

if len(df) > 0:
    df = df.sort_values(by="auc_macro", ascending=False)

print("\nFINAL PREDICTED CONTEXT RECOMMENDER RESULTS:")
print(df)

output_csv = "pred_context_recommender_with_context_head_analysis.csv"
df.to_csv(output_csv, index=False)

print("\nSaved:")
print(output_csv)
print(f"Per-antibiotic context head analyses saved in: {CONTEXT_ANALYSIS_DIR}/")