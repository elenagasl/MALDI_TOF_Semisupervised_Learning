import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import sys
import os
from sklearn.model_selection import KFold
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.recommender_dataset import RecommenderDataset
from models import AMRModel


# ============================================================
# Utils
# ============================================================

def safe_binary_auc(y_true, y_score):
    mask = ~np.isnan(y_true) & ~np.isnan(y_score)

    if np.sum(mask) == 0:
        return np.nan

    y_true = y_true[mask]
    y_score = y_score[mask]

    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, y_score)


def safe_macro_multilabel_auc(y_true, y_score):
    aucs = []
    for j in range(y_true.shape[1]):
        auc = safe_binary_auc(y_true[:, j], y_score[:, j])
        aucs.append(auc)

    valid = [a for a in aucs if not np.isnan(a)]
    return np.mean(valid), aucs


def safe_micro_auc(y_true, y_score):
    mask = ~np.isnan(y_true) & ~np.isnan(y_score)

    y_true = y_true[mask]
    y_score = y_score[mask]

    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, y_score)


def reconstruct_matrix(logits, labels, drugs, locs, n_samples, n_antibiotics):

    print("Reconstructing prediction matrix...", flush=True)

    pred = np.full((n_samples, n_antibiotics), np.nan)
    true = np.full((n_samples, n_antibiotics), np.nan)

    for l, y, d, loc in zip(logits, labels, drugs, locs):
        if not np.isnan(y):
            pred[loc, d] = 1 / (1 + np.exp(-l))
            true[loc, d] = y

    print("Matrix reconstruction DONE", flush=True)
    return pred, true


# ============================================================
# FOLDS
# ============================================================

def create_sample_folds(sample_ids, n_splits=5, seed=42):
    print("Creating folds...", flush=True)

    unique_samples = np.unique(sample_ids)
    print(f"Unique samples: {len(unique_samples)}", flush=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for i, (train_idx, test_idx) in enumerate(kf.split(unique_samples)):
        print(f"Fold split {i}: train={len(train_idx)} test={len(test_idx)}", flush=True)

        train_samples = unique_samples[train_idx]
        test_samples = unique_samples[test_idx]
        folds.append((train_samples, test_samples))

    return folds


# ============================================================
# TRAINING
# ============================================================

def train_model(model, train_loader, val_loader, device, max_epochs=1200, patience=50):

    print("Starting TRAINING...", flush=True)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(max_epochs):

        print(f"\nEpoch {epoch+1} START", flush=True)

        model.train()
        train_loss = 0.0

        for i, batch in enumerate(train_loader):

            if i == 0:
                print("First TRAIN batch", flush=True)

            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            batch["drug"] = batch["drug"].long()

            logits = model(batch)
            labels = batch["label"]

            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        print("Training loop finished", flush=True)

        # VALIDATION
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for i, batch in enumerate(val_loader):

                if i == 0:
                    print("First VAL batch", flush=True)

                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                batch["drug"] = batch["drug"].long()

                logits = model(batch)
                labels = batch["label"]

                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"[Epoch {epoch+1}] Train: {train_loss:.4f} | Val: {val_loss:.4f}", flush=True)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
            print("New BEST model", flush=True)
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})", flush=True)

        if patience_counter >= patience:
            print(f"EARLY STOPPING at epoch {epoch+1}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("Training FINISHED", flush=True)


# ============================================================
# PREDICT
# ============================================================

def predict_model(model, loader, device):

    print("Starting PREDICTION...", flush=True)

    model.eval()

    all_logits, all_labels, all_drugs, all_locs = [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):

            if i == 0:
                print("First TEST batch", flush=True)

            batch_gpu = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            batch_gpu["drug"] = batch_gpu["drug"].long()

            logits = model(batch_gpu).cpu().numpy()

            all_logits.append(logits)
            all_labels.append(batch["label"].numpy())
            all_drugs.append(batch["drug"].numpy())
            all_locs.append(batch["loc"])

    print("Prediction DONE", flush=True)

    return (
        np.concatenate(all_logits),
        np.concatenate(all_labels),
        np.concatenate(all_drugs),
        np.concatenate(all_locs),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("===== SCRIPT START =====", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("Loading dataset...", flush=True)

    with open("/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS.pkl", "rb") as f:
        payload = pickle.load(f)

    print("Payload loaded", flush=True)

    X = np.asarray(payload["data"], dtype=np.float32)
    Y = np.asarray(payload["amr"], dtype=np.float32)
    species = np.asarray(payload["label"])

    print("Shapes:", X.shape, Y.shape, flush=True)

    mask = np.isfinite(X).all(axis=1)
    X, Y, species = X[mask], Y[mask], species[mask]

    print("After cleaning:", X.shape, flush=True)

    unique_species = np.unique(species)
    print(f"Total species: {len(unique_species)}", flush=True)

    results = []

    for sp in unique_species:

        print("\n==============================", flush=True)
        print(f"Species: {sp}", flush=True)
        print("==============================", flush=True)

        idx = species == sp
        X_sp, Y_sp = X[idx], Y[idx]

        print(f"Samples: {len(X_sp)}", flush=True)

        if len(X_sp) < 50:
            print("Skipping species (too few samples)", flush=True)
            continue

        # FILTRADO
        valid_cols = []

        for j in range(Y_sp.shape[1]):
            col = Y_sp[:, j]
            col = col[~np.isnan(col)]

            if len(col) > 50 and len(np.unique(col)) >= 2:
                valid_cols.append(j)

        Y_sp = Y_sp[:, valid_cols]
        A_sp = Y_sp.shape[1]

        print(f"Valid antibiotics: {A_sp}", flush=True)

        folds = create_sample_folds(np.arange(len(X_sp)), n_splits=5)

        auc_micro_folds = []
        auc_macro_folds = []

        for fold_id, (train_samples, test_samples) in enumerate(folds):

            print(f"\n--- Fold {fold_id} ---", flush=True)

            X_train, X_test = X_sp[train_samples], X_sp[test_samples]
            Y_train, Y_test = Y_sp[train_samples], Y_sp[test_samples]

            val_split = int(0.8 * len(X_train))

            X_tr, X_val = X_train[:val_split], X_train[val_split:]
            Y_tr, Y_val = Y_train[:val_split], Y_train[val_split:]

            print("Creating DataLoaders...", flush=True)

            train_loader = DataLoader(RecommenderDataset(X_tr, Y_tr), batch_size=128, shuffle=True)
            val_loader   = DataLoader(RecommenderDataset(X_val, Y_val), batch_size=128, shuffle=False)
            test_loader  = DataLoader(RecommenderDataset(X_test, Y_test), batch_size=128, shuffle=False)

            print("Initializing model...", flush=True)

            model = AMRModel(
                spectrum_embedder="mlp",
                drug_embedder="onehot",
                spectrum_kwargs={"n_inputs": X.shape[1], "n_outputs": 64},
                drug_kwargs={"num_drugs": A_sp, "dim": 64},
            )

            train_model(model, train_loader, val_loader, device)

            logits, labels, drugs, locs = predict_model(model, test_loader, device)

            pred_mat, true_mat = reconstruct_matrix(
                logits, labels, drugs, locs,
                n_samples=len(X_test),
                n_antibiotics=A_sp
            )

            auc_macro, _ = safe_macro_multilabel_auc(true_mat, pred_mat)
            auc_micro = safe_micro_auc(true_mat, pred_mat)

            print(f"Fold {fold_id} RESULTS → micro: {auc_micro:.4f} | macro: {auc_macro:.4f}", flush=True)

            auc_micro_folds.append(auc_micro)
            auc_macro_folds.append(auc_macro)

        results.append({
            "species": sp,
            "auc_micro_mean": np.nanmean(auc_micro_folds),
            "auc_macro_mean": np.nanmean(auc_macro_folds),
            "auc_micro_std": np.nanstd(auc_micro_folds),
            "auc_macro_std": np.nanstd(auc_macro_folds),
        })

    print("\n===== FINAL RESULTS =====", flush=True)

    for r in results:
        print(r, flush=True)

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="auc_macro_mean", ascending=False)

    print("Saving CSV...", flush=True)
    df_results.to_csv("results_code1.csv", index=False)

    print(df_results, flush=True)

    print("===== SCRIPT END =====", flush=True)


if __name__ == "__main__":
    main()