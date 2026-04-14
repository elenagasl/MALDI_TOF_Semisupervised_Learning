import pickle
import numpy as np

# =========================================================
# PATHS
# =========================================================
pickle1_path = "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/MARISMa_study_MARISMA_samples.pkl"
pickle2_path = "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/DRIAMS_A_AMR_DRIAMS_ABC_samples.pkl"

output_path = "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl"

# =========================================================
# LOAD PICKLES
# =========================================================
print("Loading MARISMa...")
with open(pickle1_path, "rb") as f:
    p1 = pickle.load(f)

print("Loading DRIAMS...")
with open(pickle2_path, "rb") as f:
    p2 = pickle.load(f)

# =========================================================
# DATA
# =========================================================
X1, y1 = p1["data"], p1["label"]
X2, y2 = p2["data"], p2["label"]

# =========================================================
# AMR
# =========================================================
amr1 = p1["amr"]
ab1 = list(p1["antibiotics"])

amr2 = p2["amr"]
ab2 = list(p2["antibiotics"])

# =========================================================
# HOSPITAL (CORRECTO)
# =========================================================

# --- MARISMa (escalar → vector)
hospital1_val = p1.get("hospital", 3)
hospital1 = np.array([hospital1_val] * X1.shape[0])

# --- DRIAMS (desde meta)
if "meta" not in p2:
    raise ValueError("DRIAMS pickle has no meta field → cannot extract hospital")

hospital2 = np.array([m.get("hospital_id", -1) for m in p2["meta"]])

# =========================================================
# SAMPLE TYPE
# =========================================================

# --- MARISMa
sample_type1 = p1.get("sample_type", None)
if sample_type1 is None:
    sample_type1 = np.array([None] * X1.shape[0])

# --- DRIAMS (no existe → rellenar)
sample_type2 = np.array([None] * X2.shape[0])

# =========================================================
# DEBUG INFO
# =========================================================
print("Dataset 1 samples:", X1.shape[0])
print("Dataset 2 samples:", X2.shape[0])

print("Hospital1 unique:", np.unique(hospital1))
print("Hospital2 unique:", np.unique(hospital2))

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
# CONCATENATE EVERYTHING
# =========================================================
print("Concatenating datasets...")

X = np.vstack([X1, X2])
y = np.concatenate([y1, y2])
amr = np.vstack([amr1_new, amr2_new])

hospital = np.concatenate([hospital1, hospital2])
sample_type = np.concatenate([sample_type1, sample_type2])

# =========================================================
# FINAL CHECKS
# =========================================================
assert X.shape[0] == y.shape[0] == amr.shape[0] == hospital.shape[0] == sample_type.shape[0]

print("\nFinal dataset")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])
print("AMR shape:", amr.shape)

print("Hospital shape:", hospital.shape)
print("Sample type shape:", sample_type.shape)

# =========================================================
# SAVE NEW PICKLE
# =========================================================
payload = {
    "data": X,
    "label": y,
    "amr": amr,
    "antibiotics": all_antibiotics,
    "hospital": hospital,
    "sample_type": sample_type
}

with open(output_path, "wb") as f:
    pickle.dump(payload, f)

print("\nCombined pickle saved to:")
print(output_path)