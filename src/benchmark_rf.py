#!/usr/bin/env python3
# ==========================================================
# Benchmark 1 – Random Forest Replication (80/20 Stratified)
# Single split, same test set for Binary vs LPS
# With live progress prints (BayesSearchCV callback + flush)
# ==========================================================

import os
import sys
import pickle
import datetime
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, make_scorer
from sklearn.ensemble import RandomForestClassifier

from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

warnings.filterwarnings("ignore")


# ==========================================================
# Configuration
# ==========================================================

PICKLE_PATH = "../data/DRIAMS_A_AMR_paper_replication.pkl"
OUTPUT_DIR = "results_benchmark1_single_split"

TEST_SIZE = 0.20
RANDOM_STATE_SPLIT = 42

N_CV = 5
N_ITER = 200
N_JOBS = 10
N_POINTS = 1

# OJO: aquí he dejado el search space como el del github que tú pegaste:
# n_estimators: 1..1000, max_depth: 1..10, min_samples_leaf: 1..10
# Si quieres el del paper (100..1000, max_depth 100..101, etc.) lo cambiamos en 10 segundos.
SEARCH_SPACE = {
    "n_estimators": Integer(1, 1000),
    "max_depth": Integer(1, 10),
    "min_samples_leaf": Integer(1, 10),
    "bootstrap": Categorical([False, True]),
    "random_state": Categorical([0]),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
summary_file = os.path.join(OUTPUT_DIR, f"summary_{timestamp}.txt")


# ==========================================================
# Utility Functions
# ==========================================================

def multilabel_f1_wrapper(true: np.ndarray, pred: np.ndarray, average: str = "weighted") -> float:
    """Average per-label F1 (same style as your github code)."""
    total = 0.0
    for col in range(true.shape[1]):
        total += f1_score(true[:, col], pred[:, col], average=average)
    return total / true.shape[1]


def make_bayes_callback(tag: str):
    """
    Callback for skopt BayesSearchCV.
    It runs after each iteration and prints best score so far.
    """
    def _cb(res):
        # res is an OptimizeResult
        it = len(res.x_iters)
        best = float(np.max(res.func_vals)) if hasattr(res, "func_vals") else None
        # In BayesSearchCV, objective is maximized internally via scoring, but
        # skopt stores values as -score sometimes depending on setup.
        # We'll print iteration count only; final best is printed from opt.best_score_.
        print(f"[{tag}] Bayes iter {it}/{N_ITER} done.", flush=True)
        return False
    return _cb


def optimize_rf_exact(train_X: np.ndarray, train_y: np.ndarray, scoring, tag: str) -> BayesSearchCV:
    """
    Bayesian optimization wrapper, with live progress prints.
    """
    opt = BayesSearchCV(
        estimator=RandomForestClassifier(),
        search_spaces=SEARCH_SPACE,
        n_iter=N_ITER,
        cv=N_CV,
        random_state=0,
        n_jobs=N_JOBS,
        n_points=N_POINTS,
        verbose=0,  # dejamos verbose=0 y controlamos nosotros el log (más limpio y fiable en nohup)
        refit=True
    )

    opt.scoring = scoring

    print(f"[{tag}] BayesSearchCV starting: n_iter={N_ITER}, cv={N_CV}, n_jobs={N_JOBS}, n_points={N_POINTS}", flush=True)
    opt.fit(train_X, train_y, callback=[make_bayes_callback(tag)])
    print(f"[{tag}] BayesSearchCV finished. best_score={opt.best_score_}", flush=True)
    print(f"[{tag}] Best params: {opt.best_params_}", flush=True)

    return opt


# ==========================================================
# Main
# ==========================================================

def main():
    # Make stdout line-buffered (helps when not using -u)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    print("Loading pickle...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        payload = pickle.load(f)

    X_raw = payload["data"]
    amr_raw = payload["amr"]
    antibiotics = payload["antibiotics"]
    labels_raw = payload["label"]

    df_features = pd.DataFrame(X_raw)
    df_amr = pd.DataFrame(amr_raw, columns=antibiotics)
    df_species = pd.DataFrame(labels_raw, columns=["species"])

    full_df = pd.concat([df_features, df_amr, df_species], axis=1)

    # Remove duplicates (features + AMR + species together)
    full_df = full_df.drop_duplicates()
    print(f"Total samples after duplicate removal: {full_df.shape[0]}", flush=True)

    # Species-specific antibiotic panels (matching your pickle species names)
    species_antibiotics = {
        "Staphylococcus_Aureus": ["Oxacillin", "Clindamycin", "Fusidic acid"],
        "Escherichia_Coli": ["Ciprofloxacin", "Ceftriaxone", "Piperacillin-Tazobactam", "Cefepime"],
        "Klebsiella_Pneumoniae": ["Ciprofloxacin", "Ceftriaxone", "Imipenem", "Meropenem"],
        "Pseudomonas_Aeruginosa": ["Ciprofloxacin", "Imipenem", "Meropenem"],
    }

    results_all = {}

    n_features = X_raw.shape[1]
    feature_cols = list(full_df.columns[:n_features])

    for species, ab_list in species_antibiotics.items():
        print("\n===================================", flush=True)
        print(f"Processing species: {species}", flush=True)
        print("===================================", flush=True)

        df_species_subset = full_df[full_df["species"] == species]
        if df_species_subset.shape[0] == 0:
            print(f"[WARN] No samples found for {species}. Skipping.", flush=True)
            continue

        # Keep only features + this species antibiotics
        species_df = df_species_subset[feature_cols + ab_list].copy()

        # Drop rows with NaN in either features or these antibiotics
        before = species_df.shape[0]
        species_df = species_df.dropna()
        after = species_df.shape[0]
        print(f"Drop NaN: {before} -> {after}", flush=True)

        X = species_df.iloc[:, :n_features].to_numpy()
        y_multi = species_df[ab_list].to_numpy().astype(int)

        # Build LPS patterns
        patterns = np.array(["".join(map(str, row)) for row in y_multi])
        counts = pd.Series(patterns).value_counts()

        # Remove rare patterns (<10)
        valid_patterns = counts[counts >= 10].index
        mask_valid = np.isin(patterns, valid_patterns)

        X = X[mask_valid]
        y_multi = y_multi[mask_valid]
        patterns = patterns[mask_valid]

        print(f"Final samples: {X.shape[0]}", flush=True)
        print(f"Unique patterns: {len(np.unique(patterns))}", flush=True)
        print(f"Antibiotics ({len(ab_list)}): {ab_list}", flush=True)

        if X.shape[0] == 0:
            print(f"[WARN] No samples left after rare-pattern filtering for {species}. Skipping.", flush=True)
            continue

        # ======================================================
        # 80/20 Stratified Split by patterns
        # ======================================================
        train_idx, test_idx = train_test_split(
            np.arange(X.shape[0]),
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE_SPLIT,
            stratify=patterns,
            shuffle=True
        )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train_multi, y_test_multi = y_multi[train_idx], y_multi[test_idx]
        y_train_patterns = patterns[train_idx]

        print(f"Train size: {X_train.shape[0]}", flush=True)
        print(f"Test size : {X_test.shape[0]}", flush=True)

        # Standardize (fit only on train)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ======================================================
        # 1) Independent Binary Models
        # ======================================================
        print("\nTraining independent binary classifiers...", flush=True)
        binary_predictions = []

        for ab_idx in range(y_train_multi.shape[1]):
            ab_name = ab_list[ab_idx]
            tag = f"{species} | BIN | {ab_name}"

            print(f"\nAntibiotic {ab_idx+1}/{y_train_multi.shape[1]}: {ab_name}", flush=True)

            y_train_bin = y_train_multi[:, ab_idx]

            opt = optimize_rf_exact(
                X_train, y_train_bin,
                scoring="f1_weighted",
                tag=tag
            )

            best_model = opt.best_estimator_
            # refit=True already refits on all train, but we keep this explicit:
            best_model.fit(X_train, y_train_bin)

            pred_bin = best_model.predict(X_test)
            binary_predictions.append(pred_bin)

        binary_predictions = np.array(binary_predictions).T

        acc_bin = accuracy_score(y_test_multi, binary_predictions)
        ham_bin = hamming_loss(y_test_multi, binary_predictions)
        f1_bin = multilabel_f1_wrapper(y_test_multi, binary_predictions)

        print("\nBinary Results:", flush=True)
        print(f"Accuracy: {acc_bin}", flush=True)
        print(f"Hamming : {ham_bin}", flush=True)
        print(f"F1      : {f1_bin}", flush=True)

        # ======================================================
        # 2) LPS Model
        # ======================================================
        print("\nTraining LPS classifier...", flush=True)

        le = LabelEncoder()
        y_train_lps = le.fit_transform(y_train_patterns)

        def lps_scorer(true_lps, pred_lps):
            true_multi = np.array([[int(c) for c in s] for s in le.inverse_transform(true_lps)])
            pred_multi = np.array([[int(c) for c in s] for s in le.inverse_transform(pred_lps)])
            return multilabel_f1_wrapper(true_multi, pred_multi)

        opt_lps = optimize_rf_exact(
            X_train, y_train_lps,
            scoring=make_scorer(lps_scorer),
            tag=f"{species} | LPS"
        )

        best_lps_model = opt_lps.best_estimator_
        best_lps_model.fit(X_train, y_train_lps)

        pred_lps = best_lps_model.predict(X_test)
        decoded = le.inverse_transform(pred_lps)
        pred_multi = np.array([[int(c) for c in s] for s in decoded])

        acc_lps = accuracy_score(y_test_multi, pred_multi)
        ham_lps = hamming_loss(y_test_multi, pred_multi)
        f1_lps = multilabel_f1_wrapper(y_test_multi, pred_multi)

        print("\nLPS Results:", flush=True)
        print(f"Accuracy: {acc_lps}", flush=True)
        print(f"Hamming : {ham_lps}", flush=True)
        print(f"F1      : {f1_lps}", flush=True)

        results_all[species] = {
            "binary": (acc_bin, ham_bin, f1_bin),
            "lps": (acc_lps, ham_lps, f1_lps)
        }

    # ==========================================================
    # Save Summary
    # ==========================================================
    with open(summary_file, "w") as f:
        for species, results in results_all.items():
            f.write("\n===================================\n")
            f.write(f"Species: {species}\n")
            f.write("===================================\n")

            f.write("\n--- Independent Binary ---\n")
            f.write(f"Accuracy: {results['binary'][0]}\n")
            f.write(f"Hamming : {results['binary'][1]}\n")
            f.write(f"F1      : {results['binary'][2]}\n")

            f.write("\n--- LPS ---\n")
            f.write(f"Accuracy: {results['lps'][0]}\n")
            f.write(f"Hamming : {results['lps'][1]}\n")
            f.write(f"F1      : {results['lps'][2]}\n")

    print("\nFinished successfully.", flush=True)
    print("Results saved in:", summary_file, flush=True)


if __name__ == "__main__":
    main()