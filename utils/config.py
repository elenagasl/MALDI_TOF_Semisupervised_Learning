from pathlib import Path
# Shared dataset root (read-only)
DATASET_ROOT = Path("/export/data_ml4ds/bacteria_id")

# MARISMa structured dataset
MARISMA_ROOT = DATASET_ROOT / "MARISMa"

# Base anonymized pickle (shared)
MARISMA_ANON_PICKLE = DATASET_ROOT / "codigoMALDIVAS" / "pickles" / "MARISMa_anonymized.pkl"

# Your user root
USER_ROOT = Path("/export/usuarios01/egarroyo")

# Project root
PROJECT_ROOT = USER_ROOT / "MALDI_for_AMR_prediction"

# Output directory for your generated pickles
PICKLE_OUTPUT_DIR = PROJECT_ROOT / "data"

# Klebsiella pickle
KLEBSIELLA_PICKLE = PICKLE_OUTPUT_DIR / "MARISMa_study_Klebsiella_AMR.pkl"

# Driams A pickle
DRIAMS_A_PICKLE =  PICKLE_OUTPUT_DIR / "DRIAMS_A_AMR_paper_replication.pkl"
