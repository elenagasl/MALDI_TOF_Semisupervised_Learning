import pickle
import numpy as np
import pandas as pd

# ============================================================
# PATH
# ============================================================
DATA_PATH = "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl"

# ============================================================
# LOAD DATA
# ============================================================
print("Loading dataset...", flush=True)

with open(DATA_PATH, "rb") as f:
    payload = pickle.load(f)

X = payload["data"]
y_species = payload["label"]
amr = payload["amr"]
antibiotics = list(payload["antibiotics"])

print("Data loaded!", flush=True)

# ============================================================
# BUILD PAIR COUNTS
# ============================================================
pair_counts = {}

n_samples = len(y_species)
n_antibiotics = len(antibiotics)

for i in range(n_samples):

    species = y_species[i]

    for j in range(n_antibiotics):

        value = amr[i, j]

        # Filtrar valores inválidos
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        if value == -1:
            continue

        antibiotic = antibiotics[j]

        key = (species, antibiotic)

        if key not in pair_counts:
            pair_counts[key] = 0

        pair_counts[key] += 1

# ============================================================
# CONVERT TO DATAFRAME
# ============================================================
df_pairs = pd.DataFrame([
    {"species": k[0], "antibiotic": k[1], "count": v}
    for k, v in pair_counts.items()
])

# ============================================================
# SORT BY FREQUENCY
# ============================================================
df_pairs = df_pairs.sort_values(by="count", ascending=False).reset_index(drop=True)

# ============================================================
# FILTER (opcional)
# ============================================================
MIN_COUNT = 50  # 🔥 ajusta esto

df_filtered = df_pairs[df_pairs["count"] >= MIN_COUNT].reset_index(drop=True)

# ============================================================
# OUTPUT
# ============================================================
print("\nTop pairs:")
print(df_filtered.head(20))

print("\nTotal pairs (all):", len(df_pairs))
print("Pairs with >=", MIN_COUNT, "samples:", len(df_filtered))

# ============================================================
# SAVE
# ============================================================
df_pairs.to_csv("all_pairs_counts.csv", index=False)
df_filtered.to_csv("filtered_pairs_counts.csv", index=False)

print("\nSaved:")
print(" - all_pairs_counts.csv")
print(" - filtered_pairs_counts.csv")