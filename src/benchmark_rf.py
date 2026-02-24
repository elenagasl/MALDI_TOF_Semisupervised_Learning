#!/usr/bin/env python3

# ==========================================================
# Benchmark 1 – Random Forest Replication (Scientific Reports 2024)
# ==========================================================

import pickle
import numpy as np
import pandas as pd
import warnings
import os
import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    hamming_loss,
    make_scorer
)
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

warnings.filterwarnings("ignore")

# ==========================================================
# Configuration
# ==========================================================

PICKLE_PATH = "../data/DRIAMS_A_AMR_paper_replication.pkl"
OUTPUT_DIR = "results_benchmark1"
N_OUTER = 10
N_CV = 5
N_ITER = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
summary_file = os.path.join(OUTPUT_DIR, f"summary_{timestamp}.txt")

# ==========================================================
# Utility Functions
# ==========================================================

def multilabel_f1_wrapper(true, pred, average="weighted"):
    total = 0
    for col in range(true.shape[1]):
        total += f1_score(true[:, col], pred[:, col], average=average)
    return total / true.shape[1]


def optimize_rf_exact(train_X, train_y, scoring):

    search_space = {
        "n_estimators": Integer(1, 1000),
        "max_depth": Integer(1, 10),
        "min_samples_leaf": Integer(1, 10),
        "bootstrap": Categorical([False, True]),
        "random_state": Categorical([0])
    }

    opt = BayesSearchCV(
        RandomForestClassifier(),
        search_space,
        n_iter=N_ITER,
        cv=N_CV,
        random_state=0,
        n_jobs=10,
        n_points=2,
        verbose=0
    )

    opt.scoring = scoring
    opt.fit(train_X, train_y)

    return opt


# ==========================================================
# Load Data
# ==========================================================

print("Loading pickle...")

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

# Remove duplicates
full_df = full_df.drop_duplicates()

# ==========================================================
# Species-specific antibiotic panels
# ==========================================================

species_antibiotics = {
    "Staphylococcus_Aureus": [
        "Oxacillin", "Clindamycin", "Fusidic acid"
    ],
    "Escherichia_Coli": [
        "Ciprofloxacin", "Ceftriaxone",
        "Piperacillin-Tazobactam", "Cefepime"
    ],
    "Klebsiella_Pneumoniae": [
        "Ciprofloxacin", "Ceftriaxone",
        "Imipenem", "Meropenem"
    ],
    "Pseudomonas_Aeruginosa": [
        "Ciprofloxacin", "Imipenem", "Meropenem"
    ]
}

outer_results = {}

# ==========================================================
# MAIN LOOP PER SPECIES
# ==========================================================

for species, ab_list in species_antibiotics.items():

    print("\n===================================")
    print("Processing species:", species)
    print("===================================")

    df_species_subset = full_df[full_df["species"] == species]

    feature_cols = full_df.columns[:X_raw.shape[1]]
    species_df = df_species_subset[list(feature_cols) + ab_list]

    species_df = species_df.dropna()

    X = species_df.iloc[:, :X_raw.shape[1]].values
    y_multi = species_df[ab_list].values

    # LPS patterns
    patterns = np.array([
        "".join(map(str, row.astype(int)))
        for row in y_multi
    ])

    # Remove rare patterns
    counts = pd.Series(patterns).value_counts()
    valid_patterns = counts[counts > 10].index
    mask_valid = np.isin(patterns, valid_patterns)

    X = X[mask_valid]
    y_multi = y_multi[mask_valid]
    patterns = patterns[mask_valid]

    print("Final samples:", X.shape[0])

    outer_cv = StratifiedKFold(
        n_splits=N_OUTER,
        shuffle=True,
        random_state=42
    )

    results_binary = []
    results_lps = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, patterns)):

        print(f"Outer Fold {fold+1}/{N_OUTER}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train_multi, y_test_multi = y_multi[train_idx], y_multi[test_idx]
        y_train_patterns = patterns[train_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # =============================
        # Independent binary models
        # =============================

        binary_predictions = []

        for ab_idx in range(y_train_multi.shape[1]):

            y_train_bin = y_train_multi[:, ab_idx]

            opt = optimize_rf_exact(
                X_train,
                y_train_bin,
                scoring="f1_weighted"
            )

            best_model = opt.best_estimator_
            best_model.fit(X_train, y_train_bin)

            pred_bin = best_model.predict(X_test)
            binary_predictions.append(pred_bin)

        binary_predictions = np.array(binary_predictions).T

        acc_bin = accuracy_score(y_test_multi, binary_predictions)
        ham_bin = hamming_loss(y_test_multi, binary_predictions)
        f1_bin = multilabel_f1_wrapper(y_test_multi, binary_predictions)

        results_binary.append((acc_bin, ham_bin, f1_bin))

        # =============================
        # LPS model
        # =============================

        le = LabelEncoder()
        y_train_lps = le.fit_transform(y_train_patterns)

        opt = optimize_rf_exact(
            X_train,
            y_train_lps,
            scoring=make_scorer(
                lambda t, p: multilabel_f1_wrapper(
                    np.array([[int(c) for c in s] for s in le.inverse_transform(t)]),
                    np.array([[int(c) for c in s] for s in le.inverse_transform(p)])
                )
            )
        )

        best_model = opt.best_estimator_
        best_model.fit(X_train, y_train_lps)

        pred_lps = best_model.predict(X_test)

        decoded = le.inverse_transform(pred_lps)
        pred_multi = np.array([[int(c) for c in s] for s in decoded])

        acc_lps = accuracy_score(y_test_multi, pred_multi)
        ham_lps = hamming_loss(y_test_multi, pred_multi)
        f1_lps = multilabel_f1_wrapper(y_test_multi, pred_multi)

        results_lps.append((acc_lps, ham_lps, f1_lps))

    outer_results[species] = {
        "binary_mean": np.mean(results_binary, axis=0),
        "binary_std": np.std(results_binary, axis=0),
        "lps_mean": np.mean(results_lps, axis=0),
        "lps_std": np.std(results_lps, axis=0)
    }

# ==========================================================
# Save Summary
# ==========================================================

with open(summary_file, "w") as f:

    for species, results in outer_results.items():

        f.write("\n===================================\n")
        f.write(f"Species: {species}\n")
        f.write("===================================\n")

        f.write("\n--- Independent Binary ---\n")
        f.write(f"Accuracy: {results['binary_mean'][0]} ± {results['binary_std'][0]}\n")
        f.write(f"Hamming : {results['binary_mean'][1]} ± {results['binary_std'][1]}\n")
        f.write(f"F1      : {results['binary_mean'][2]} ± {results['binary_std'][2]}\n")

        f.write("\n--- LPS ---\n")
        f.write(f"Accuracy: {results['lps_mean'][0]} ± {results['lps_std'][0]}\n")
        f.write(f"Hamming : {results['lps_mean'][1]} ± {results['lps_std'][1]}\n")
        f.write(f"F1      : {results['lps_mean'][2]} ± {results['lps_std'][2]}\n")

print("\nFinished successfully.")
print("Results saved in:", summary_file)