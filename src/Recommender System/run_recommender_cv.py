import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import sys
import os
import time
from sklearn.model_selection import ShuffleSplit

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


def reconstruct_matrix(logits, labels, drugs, locs, n_samples, n_antibiotics):
    pred = np.full((n_samples, n_antibiotics), np.nan)
    true = np.full((n_samples, n_antibiotics), np.nan)

    for l, y, d, loc in zip(logits, labels, drugs, locs):
        if not np.isnan(y):
            pred[loc, d] = 1 / (1 + np.exp(-l))
            true[loc, d] = y

    return pred, true


# ============================================================
# Training con Early Stopping
# ============================================================

def train_model(model, train_loader, val_loader, device, max_epochs=1200, patience=50):

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(max_epochs):

        # -------- TRAIN --------
        model.train()
        train_loss = 0.0

        for batch in train_loader:
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

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                batch["drug"] = batch["drug"].long()

                logits = model(batch)
                labels = batch["label"]

                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # -------- EARLY STOPPING --------
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # cargar mejor modelo
    if best_state is not None:
        model.load_state_dict(best_state)


# ============================================================
# Predict
# ============================================================

def predict_model(model, loader, device):
    model.eval()

    all_logits, all_labels, all_drugs, all_locs = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            batch_gpu = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            batch_gpu["drug"] = batch_gpu["drug"].long()

            logits = model(batch_gpu).cpu().numpy()

            all_logits.append(logits)
            all_labels.append(batch["label"].numpy())
            all_drugs.append(batch["drug"].numpy())
            all_locs.append(batch["loc"])

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS.pkl", "rb") as f:
        payload = pickle.load(f)

    X = np.asarray(payload["data"], dtype=np.float32)
    Y = np.asarray(payload["amr"], dtype=np.float32)
    species = np.asarray(payload["label"])

    mask = np.isfinite(X).all(axis=1)
    X, Y, species = X[mask], Y[mask], species[mask]

    unique_species = np.unique(species)
    results = []

    for sp in unique_species:

        print("\n==============================")
        print(f"Species: {sp}")
        print("==============================")

        idx = species == sp
        X_sp, Y_sp = X[idx], Y[idx]

        if len(X_sp) < 50:
            continue

        # filtrar antibióticos
        valid_cols = []
        for j in range(Y_sp.shape[1]):
            col = Y_sp[:, j]
            col = col[~np.isnan(col)]
            if len(np.unique(col)) >= 2:
                valid_cols.append(j)

        Y_sp = Y_sp[:, valid_cols]
        A_sp = Y_sp.shape[1]

        splitter = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
        fold_aucs = []

        for fold, (train_idx, test_idx) in enumerate(splitter.split(X_sp)):

            print(f"\n--- Fold {fold} ---")

            X_train, X_test = X_sp[train_idx], X_sp[test_idx]
            Y_train, Y_test = Y_sp[train_idx], Y_sp[test_idx]

            # split train → train/val (80/20)
            val_split = int(0.8 * len(X_train))

            X_tr, X_val = X_train[:val_split], X_train[val_split:]
            Y_tr, Y_val = Y_train[:val_split], Y_train[val_split:]

            train_loader = DataLoader(RecommenderDataset(X_tr, Y_tr), batch_size=128, shuffle=True)
            val_loader   = DataLoader(RecommenderDataset(X_val, Y_val), batch_size=128, shuffle=False)
            test_loader  = DataLoader(RecommenderDataset(X_test, Y_test), batch_size=128, shuffle=False)

            model = AMRModel(
                spectrum_embedder="mlp",
                drug_embedder="onehot",
                spectrum_kwargs={"n_inputs": X.shape[1], "n_outputs": 64},
                drug_kwargs={"num_drugs": A_sp, "dim": 64},
            )

            train_model(model, train_loader, val_loader, device, max_epochs=1200, patience=50)

            logits, labels, drugs, locs = predict_model(model, test_loader, device)

            pred_mat, true_mat = reconstruct_matrix(
                logits, labels, drugs, locs,
                n_samples=len(X_test),
                n_antibiotics=A_sp
            )

            auc_macro, _ = safe_macro_multilabel_auc(true_mat, pred_mat)

            print(f"AUC fold: {auc_macro:.4f}")
            fold_aucs.append(auc_macro)

        mean_auc = np.nanmean(fold_aucs)
        std_auc  = np.nanstd(fold_aucs)

        print(f"\n>>> {sp}: {mean_auc:.4f} ± {std_auc:.4f}")

        results.append({
            "species": sp,
            "auc_mean": mean_auc,
            "auc_std": std_auc,
            "n_samples": len(X_sp),
            "n_antibiotics": A_sp
        })

    print("\n\nFINAL RESULTS\n")

    for r in results:
        print(
            f"{r['species']},"
            f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f},"
            f"{r['n_samples']},"
            f"{r['n_antibiotics']}"
        )


if __name__ == "__main__":
    main()