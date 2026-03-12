import pickle
import numpy as np

# =========================================================
# PATHS
# =========================================================
pickle1_path = "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/MARISMa_study_MARISMA_half_pipeline.pkl"
pickle2_path = "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/DRIAMS_A_AMR_DRIAMS_ABC.pkl"

output_path = "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS.pkl"

# =========================================================
# LOAD PICKLES
# =========================================================
print("Loading pickle 1...")
with open(pickle1_path, "rb") as f:
    p1 = pickle.load(f)

print("Loading pickle 2...")
with open(pickle2_path, "rb") as f:
    p2 = pickle.load(f)

X1 = p1["data"]
y1 = p1["label"]
amr1 = p1["amr"]
ab1 = list(p1["antibiotics"])

X2 = p2["data"]
y2 = p2["label"]
amr2 = p2["amr"]
ab2 = list(p2["antibiotics"])

print("Dataset 1 samples:", X1.shape[0])
print("Dataset 2 samples:", X2.shape[0])

# =========================================================
# CHECK FEATURES
# =========================================================
if X1.shape[1] != X2.shape[1]:
    raise ValueError("Spectra feature dimension mismatch")

# =========================================================
# UNION OF ANTIBIOTICS
# =========================================================
all_antibiotics = sorted(list(set(ab1) | set(ab2)))

print("Total antibiotics after merge:", len(all_antibiotics))

# mapping antibiotic -> index
ab_index = {ab: i for i, ab in enumerate(all_antibiotics)}

# =========================================================
# BUILD NEW AMR MATRICES
# =========================================================
def expand_amr_matrix(amr, antibiotics_old):

    N = amr.shape[0]
    A = len(all_antibiotics)

    new_amr = np.full((N, A), np.nan)

    for j, ab in enumerate(antibiotics_old):
        new_j = ab_index[ab]
        new_amr[:, new_j] = amr[:, j]

    return new_amr

print("Expanding AMR matrices...")

amr1_new = expand_amr_matrix(amr1, ab1)
amr2_new = expand_amr_matrix(amr2, ab2)

# =========================================================
# CONCATENATE DATASETS
# =========================================================
X = np.vstack([X1, X2])
y = np.concatenate([y1, y2])
amr = np.vstack([amr1_new, amr2_new])

print("\nFinal dataset")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])
print("AMR shape:", amr.shape)

# =========================================================
# SAVE NEW PICKLE
# =========================================================
payload = {
    "data": X,
    "label": y,
    "amr": amr,
    "antibiotics": all_antibiotics
}

with open(output_path, "wb") as f:
    pickle.dump(payload, f)

print("\nCombined pickle saved to:")
print(output_path)