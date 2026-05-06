import pickle
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from pytorch_lightning.callbacks import EarlyStopping

from lib.NCF import NCF
from lib.RecDataset import RecDataset


# =========================
# CONFIG
# =========================
DATA_PATH = '/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl'

N_SPLITS = 2
BATCH_SIZE = 64
TRAIN_BATCH_SIZE = 128
MAX_EPOCHS = 300
PATIENCE = 15

# Training context config
EMPTY_CONTEXT_PROB = 0.05
MIN_CONTEXT_FRACTION = 0.5
MAX_CONTEXT_FRACTION = 1.0

# Autoregressive inference config
AR_STEPS = 5
AR_TOP_PERCENT = 0.1
AR_THRESHOLD = 0.4

# Correlation-guided autoregression
USE_CORRELATION_GUIDANCE = True
CORRELATION_ALPHA = 1.0
MIN_CORR_PAIR = 30

# Oracle diagnostic context
ORACLE_CONTEXT_FRACTION = 0.5
ORACLE_SEED = 42

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", DEVICE)

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
# BUILD MATRIX DIRECTO
# =========================
def build_species_data(species):

    mask = (y_species_all == species)

    X = X_all[mask]
    amr = amr_all[mask].copy()

    # =========================
    # CLEAN AMR VALUES
    # =========================
    # Only binary labels are allowed:
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

    return X, amr


# =========================
# FOLDS POR SAMPLE
# =========================
def create_folds(n_samples, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(kf.split(np.arange(n_samples)))


# =========================
# PHI CORRELATION MATRIX
# =========================
def compute_phi_correlation_matrix(amr_train, min_pair=30):
    """
    Computes phi correlation between antibiotic resistance profiles.

    Only train data is used.
    NaNs are ignored pairwise.
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


def compute_antibiotic_centrality_from_phi(phi):
    """
    Global informativeness score per antibiotic.

    Uses mean absolute phi correlation.
    High value = antibiotic tends to be informative about others.
    """

    if phi is None or phi.size == 0:
        return None

    centrality = np.mean(np.abs(phi), axis=1)

    if np.max(centrality) > 0:
        centrality = centrality / np.max(centrality)

    return centrality.astype(np.float32)


# =========================
# PREDICT ALL ANTIBIOTICS
# =========================
def predict_all_antibiotics(model, maldi, amr_context, context_mask, num_items, batch_size=512):
    """
    Predicts resistance probability for every antibiotic for every sample.

    Output:
        preds_matrix: shape (N samples, num_items)
    """

    model.eval()

    N = maldi.shape[0]
    preds_matrix = np.zeros((N, num_items), dtype=np.float32)

    with torch.no_grad():

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            maldi_batch = torch.tensor(maldi[start:end]).float().to(model.device)
            amr_batch = torch.tensor(amr_context[start:end]).float().to(model.device)
            mask_batch = torch.tensor(context_mask[start:end]).float().to(model.device)

            batch_size_real = maldi_batch.shape[0]

            drug_preds = []

            for drug_id in range(num_items):

                target_drug = torch.full(
                    (batch_size_real,),
                    drug_id,
                    dtype=torch.long,
                    device=model.device
                )

                # No family info for now: dummy family = 0
                target_family = torch.zeros(
                    batch_size_real,
                    dtype=torch.long,
                    device=model.device
                )

                # Extra safety: target antibiotic must not be in its own context
                amr_tmp = amr_batch.clone()
                mask_tmp = mask_batch.clone()

                amr_tmp[:, drug_id] = 0
                mask_tmp[:, drug_id] = 0

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


# =========================
# ONE-SHOT INFERENCE
# =========================
def one_shot_inference(model, maldi, num_items):
    """
    One-shot inference:
    all AMR context is masked.
    """

    N = maldi.shape[0]

    amr_context = np.zeros((N, num_items), dtype=np.float32)
    context_mask = np.zeros((N, num_items), dtype=np.float32)

    preds = predict_all_antibiotics(
        model=model,
        maldi=maldi,
        amr_context=amr_context,
        context_mask=context_mask,
        num_items=num_items
    )

    return preds


# =========================
# ORACLE CONTEXT INFERENCE
# =========================
def oracle_context_inference(
    model,
    maldi,
    amr_true,
    num_items,
    context_fraction=0.5,
    seed=42
):
    """
    Diagnostic only.

    Uses real test AMR as partial context.
    This is NOT a valid deployment evaluation.
    It tells us whether the model can use AMR context if the context is correct.
    """

    rng = np.random.default_rng(seed)

    N = maldi.shape[0]

    amr_context = np.zeros((N, num_items), dtype=np.float32)
    context_mask = np.zeros((N, num_items), dtype=np.float32)

    for i in range(N):

        observed = np.where(~np.isnan(amr_true[i]))[0]

        if len(observed) == 0:
            continue

        n_keep = max(1, int(context_fraction * len(observed)))

        keep_idx = rng.choice(
            observed,
            size=n_keep,
            replace=False
        )

        amr_context[i, keep_idx] = amr_true[i, keep_idx]
        context_mask[i, keep_idx] = 1

    preds = predict_all_antibiotics(
        model=model,
        maldi=maldi,
        amr_context=amr_context,
        context_mask=context_mask,
        num_items=num_items
    )

    return preds


# =========================
# AUTOREGRESSIVE INFERENCE
# =========================
def autoregressive_inference(
    model,
    maldi,
    num_items,
    steps=5,
    top_percent=0.1,
    threshold=0.3,
    centrality=None,
    use_correlation_guidance=True,
    correlation_alpha=1.0
):
    """
    Autoregressive inference:
    1. Start with empty AMR context.
    2. Predict all antibiotics.
    3. Add top-k most confident and informative predictions as pseudo-context.
    4. Repeat.

    If centrality is provided, selection score becomes:
        score = confidence * (1 + alpha * centrality)
    """

    N = maldi.shape[0]

    amr_context = np.zeros((N, num_items), dtype=np.float32)
    context_mask = np.zeros((N, num_items), dtype=np.float32)

    if centrality is None:
        centrality = np.zeros(num_items, dtype=np.float32)

    centrality = np.asarray(centrality, dtype=np.float32)

    for step in range(steps):

        preds = predict_all_antibiotics(
            model=model,
            maldi=maldi,
            amr_context=amr_context,
            context_mask=context_mask,
            num_items=num_items
        )

        confidence = np.abs(preds - 0.5)

        if use_correlation_guidance:
            score = confidence * (1.0 + correlation_alpha * centrality[None, :])
        else:
            score = confidence

        k = max(1, int(top_percent * num_items))

        updates = 0

        for i in range(N):

            # Do not reselect already used positions
            available = context_mask[i] == 0

            if not np.any(available):
                continue

            score_i = score[i].copy()
            score_i[~available] = -1

            conf_i = confidence[i].copy()

            top_idx = np.argsort(score_i)[::-1][:k]

            # confidence threshold remains based on raw confidence
            valid_idx = top_idx[conf_i[top_idx] > threshold]

            if len(valid_idx) > 0:

                # Binary pseudo-context:
                # selection is based on confidence/correlation score,
                # but the value passed as context is 0/1.
                amr_context[i, valid_idx] = (preds[i, valid_idx] > 0.5).astype(np.float32)

                context_mask[i, valid_idx] = 1
                updates += len(valid_idx)

        updates_per_sample = context_mask.sum(axis=1)

        print(
            f"Autoregressive step {step + 1}/{steps} - "
            f"updates: {updates} - "
            f"context filled: {int(context_mask.sum())}/{context_mask.size} - "
            f"context/sample min={updates_per_sample.min():.0f}, "
            f"mean={updates_per_sample.mean():.2f}, "
            f"max={updates_per_sample.max():.0f}",
            flush=True
        )

        if updates == 0:
            break

    final_preds = predict_all_antibiotics(
        model=model,
        maldi=maldi,
        amr_context=amr_context,
        context_mask=context_mask,
        num_items=num_items
    )

    return final_preds


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
# TRAIN ONE SPECIES
# =========================
def train_species(species):

    print(f"\n====================")
    print(f"Species: {species}")
    print("====================", flush=True)

    data = build_species_data(species)

    if data is None:
        print("Skipping...", flush=True)
        return None, None, None

    X, amr = data

    num_items = amr.shape[1]
    n_samples = X.shape[0]

    print("Samples:", n_samples, "Items:", num_items, flush=True)

    # No family information for now
    antibiotic_families = np.zeros(num_items, dtype=int)
    num_families = 1

    folds = create_folds(n_samples, N_SPLITS)

    one_shot_micro = []
    one_shot_macro = []

    ar_micro = []
    ar_macro = []

    oracle_micro = []
    oracle_macro = []

    for fold, (train_idx, test_idx) in enumerate(folds):

        print(f"Fold {fold}", flush=True)

        val_idx = folds[(fold + 1) % N_SPLITS][1]

        X_tr, X_val, X_tst = X[train_idx], X[val_idx], X[test_idx]
        amr_tr, amr_val, amr_tst = amr[train_idx], amr[val_idx], amr[test_idx]

        # =========================
        # CORRELATION GUIDANCE FROM TRAIN ONLY
        # =========================
        phi_train = compute_phi_correlation_matrix(
            amr_train=amr_tr,
            min_pair=MIN_CORR_PAIR
        )

        centrality = compute_antibiotic_centrality_from_phi(phi_train)

        if centrality is not None:
            print(
                "Correlation centrality:",
                "min", float(np.min(centrality)),
                "mean", float(np.mean(centrality)),
                "max", float(np.max(centrality)),
                flush=True
            )

        # =========================
        # DATASETS
        # =========================
        train_sample_ids = np.arange(len(X_tr))
        val_sample_ids = np.arange(len(X_val))

        train_sample_amr = {
            i: amr_tr[i] for i in range(len(X_tr))
        }

        val_sample_amr = {
            i: amr_val[i] for i in range(len(X_val))
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
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=4,
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
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        # =========================
        # MODEL
        # =========================
        model = NCF(
            num_feat=X_tr.shape[1],
            num_items=num_items,
            num_families=num_families,
            amr_dropout=0.1
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

        # ======================
        # ONE-SHOT INFERENCE
        # ======================
        preds_one_shot = one_shot_inference(
            model=model,
            maldi=X_tst,
            num_items=num_items
        )

        micro, macro = compute_metrics(amr_tst, preds_one_shot)

        if not np.isnan(micro):
            one_shot_micro.append(micro)

        if not np.isnan(macro):
            one_shot_macro.append(macro)

        # ======================
        # ORACLE CONTEXT INFERENCE
        # ======================
        preds_oracle = oracle_context_inference(
            model=model,
            maldi=X_tst,
            amr_true=amr_tst,
            num_items=num_items,
            context_fraction=ORACLE_CONTEXT_FRACTION,
            seed=ORACLE_SEED + fold
        )

        micro, macro = compute_metrics(amr_tst, preds_oracle)

        if not np.isnan(micro):
            oracle_micro.append(micro)

        if not np.isnan(macro):
            oracle_macro.append(macro)

        # ======================
        # AUTOREGRESSIVE INFERENCE
        # ======================
        preds_ar = autoregressive_inference(
            model=model,
            maldi=X_tst,
            num_items=num_items,
            steps=AR_STEPS,
            top_percent=AR_TOP_PERCENT,
            threshold=AR_THRESHOLD,
            centrality=centrality,
            use_correlation_guidance=USE_CORRELATION_GUIDANCE,
            correlation_alpha=CORRELATION_ALPHA
        )

        micro, macro = compute_metrics(amr_tst, preds_ar)

        if not np.isnan(micro):
            ar_micro.append(micro)

        if not np.isnan(macro):
            ar_macro.append(macro)

    one_shot_result = {
        "species": species,
        "auc_micro": np.mean(one_shot_micro) if one_shot_micro else np.nan,
        "auc_macro": np.mean(one_shot_macro) if one_shot_macro else np.nan,
        "auc_micro_std": np.std(one_shot_micro) if one_shot_micro else np.nan,
        "auc_macro_std": np.std(one_shot_macro) if one_shot_macro else np.nan
    }

    ar_result = {
        "species": species,
        "auc_micro": np.mean(ar_micro) if ar_micro else np.nan,
        "auc_macro": np.mean(ar_macro) if ar_macro else np.nan,
        "auc_micro_std": np.std(ar_micro) if ar_micro else np.nan,
        "auc_macro_std": np.std(ar_macro) if ar_macro else np.nan
    }

    oracle_result = {
        "species": species,
        "auc_micro": np.mean(oracle_micro) if oracle_micro else np.nan,
        "auc_macro": np.mean(oracle_macro) if oracle_macro else np.nan,
        "auc_micro_std": np.std(oracle_micro) if oracle_micro else np.nan,
        "auc_macro_std": np.std(oracle_macro) if oracle_macro else np.nan
    }

    return one_shot_result, ar_result, oracle_result


# =========================
# MAIN
# =========================
one_shot_results = []
autoregressive_results = []
oracle_results = []

for sp in species_list:

    try:
        one_res, ar_res, oracle_res = train_species(sp)

        if one_res is not None:
            one_shot_results.append(one_res)

        if ar_res is not None:
            autoregressive_results.append(ar_res)

        if oracle_res is not None:
            oracle_results.append(oracle_res)

    except Exception as e:
        print(f"Error in {sp}:", e, flush=True)


df_one = pd.DataFrame(one_shot_results)
df_ar = pd.DataFrame(autoregressive_results)
df_oracle = pd.DataFrame(oracle_results)

if len(df_one) > 0:
    df_one = df_one.sort_values(by="auc_macro", ascending=False)

if len(df_ar) > 0:
    df_ar = df_ar.sort_values(by="auc_macro", ascending=False)

if len(df_oracle) > 0:
    df_oracle = df_oracle.sort_values(by="auc_macro", ascending=False)

print("\nFINAL ONE-SHOT RESULTS:")
print(df_one)

print("\nFINAL AUTOREGRESSIVE RESULTS:")
print(df_ar)

print("\nFINAL ORACLE CONTEXT RESULTS:")
print(df_oracle)

df_one.to_csv("one_shot.csv", index=False)
df_ar.to_csv("autoregressive.csv", index=False)
df_oracle.to_csv("oracle_context.csv", index=False)

print("\nSaved:")
print("one_shot.csv")
print("autoregressive.csv")
print("oracle_context.csv")