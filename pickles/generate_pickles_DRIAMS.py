import os
import sys
from tqdm import tqdm
import numpy as np
import pickle
import logging
from datetime import datetime
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataloader.DRIAMS_Manager import DRIAMS_Manager, DRIAMS
from utils.preprocess import (
    SequentialPreprocessor,
    VarStabilizer,
    Smoother,
    BaselineCorrecter,
    Trimmer,
    Binner,
    StdThresholder,
    LogScaler,
)
from utils.config import DATASET_ROOT, PICKLE_OUTPUT_DIR


# ==========================================================
# LOAD AMR MAP
# ==========================================================
def load_driams_amr(root_dir, antibiotics, years=None, centers=None, logger=None):

    amr_map = {}

    for center in os.listdir(root_dir):
        if centers and center not in centers:
            continue

        id_path = os.path.join(root_dir, center, "id")
        if not os.path.isdir(id_path):
            continue

        for year in os.listdir(id_path):
            if years and year not in years:
                continue

            csv_path = os.path.join(id_path, year, f"{year}_clean.csv")
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path, low_memory=False)

            for _, row in df.iterrows():
                vals = []

                for ab in antibiotics:
                    if ab not in df.columns:
                        vals.append(np.nan)
                        continue

                    v = row[ab]
                    if pd.isna(v):
                        vals.append(np.nan)
                    else:
                        s = str(v).strip().upper()
                        if s in {"R", "I"}:
                            vals.append(1.0)
                        elif s == "S":
                            vals.append(0.0)
                        else:
                            vals.append(np.nan)

                if np.all(np.isnan(vals)):
                    continue

                key = (center, year, row["code"])
                amr_map[key] = np.array(vals, dtype=float)

    if logger:
        logger.info(f"AMR samples loaded: {len(amr_map)}")

    return amr_map


# ==========================================================
# COLLECT SPECIES
# ==========================================================
def collect_species(dataset_path, preprocess_pipeline, species_list, logger,
                    amr_antibiotics=None, amr_year=None):

    logger.info("Loading DRIAMS manager...")
    manager = DRIAMS_Manager(dataset_path)

    logger.info("Querying DRIAMS_A only...")
    spectra_dict = manager.query_spectra_dict(
        centers=["DRIAMS_A"],
        genus_species=species_list
    )

    dataset = DRIAMS(
        spectra_dict,
        preprocess_pipeline=preprocess_pipeline
    )

    amr_map = None
    if amr_antibiotics:
        amr_map = load_driams_amr(
            dataset_path,
            antibiotics=amr_antibiotics,
            years=[str(amr_year)] if amr_year else None,
            centers=["DRIAMS_A"],
            logger=logger
        )

    spectra, labels, metas, amrs = [], [], [], []

    logger.info("Extracting spectra...")
    for i in tqdm(range(len(dataset)), desc="Extracting spectra"):
        spectrum_obj, label, meta = dataset[i]
        key = (meta["hospital"], meta["year"], meta["study"])

        if amr_map is not None:
            if key not in amr_map:
                continue

            amr_vec = amr_map[key]

            if np.all(np.isnan(amr_vec)):
                continue

            amrs.append(amr_vec)

        spectra.append(spectrum_obj.intensity)
        labels.append(label)
        metas.append(meta)

    data = np.stack(spectra)
    label = np.array(labels)
    meta = np.array(metas)
    amr_arr = np.stack(amrs) if amrs else None

    logger.info(f"Final data shape: {data.shape}")
    logger.info(f"Final label shape: {label.shape}")
    if amr_arr is not None:
        logger.info(f"Final AMR shape: {amr_arr.shape}")

    return data, label, meta, amr_arr


# ==========================================================
# MAIN
# ==========================================================
def main(name, preprocess_pipeline, species_list,
         amr_antibiotics=None, amr_year=None):

    log_file = os.path.join(os.path.dirname(__file__),
                            f'pickle_creation_{name}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("driams_pickle_logger")

    dataset_path = DATASET_ROOT / "relevant_datasets/10.5061/dryad.bzkh1899q/DRIAMS_ROOT"
    dataset_path = str(dataset_path)

    name_pickle = f"DRIAMS_A_AMR_{name}"

    PICKLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PICKLE_OUTPUT_DIR / f"{name_pickle}.pkl"

    logger.info("=" * 80)
    logger.info(f"Creating pickle: {name_pickle}")
    logger.info("=" * 80)

    data, label, meta, amr_arr = collect_species(
        dataset_path,
        preprocess_pipeline,
        species_list,
        logger,
        amr_antibiotics=amr_antibiotics,
        amr_year=amr_year
    )

    payload = {
        "data": data,
        "label": label,
        "meta": meta,
        "amr": amr_arr,
        "antibiotics": amr_antibiotics
    }

    with open(str(save_path), "wb") as f:
        pickle.dump(payload, f)

    logger.info(f"Pickle saved at: {save_path}")
    logger.info("=" * 80)


# ==========================================================
# EXECUTION
# ==========================================================
if __name__ == "__main__":

    name = "paper_replication"

    species_list = [
        ("Staphylococcus", "Aureus"),
        ("Escherichia", "Coli"),
        ("Klebsiella", "Pneumoniae"),
        ("Pseudomonas", "Aeruginosa"),
    ]

    amr_antibiotics = [
        # S. aureus
        "Oxacillin", "Clindamycin", "Fusidic acid",

        # E. coli
        "Ciprofloxacin", "Ceftriaxone",
        "Piperacillin-Tazobactam", "Cefepime",

        # K. pneumoniae
        "Imipenem", "Meropenem"
    ]

    preprocess_pipeline = SequentialPreprocessor(
       # VarStabilizer(method="sqrt"),
      #  Smoother(halfwindow=10),
       # BaselineCorrecter(method="SNIP", snip_n_iter=20),
      #  StdThresholder(factor=1.0),
        Trimmer(min=2000, max=10000),
        Binner(start=2000, stop=10000, step=5, aggregation="mean"),
      #  LogScaler(base=10)
    )

    main(
        name,
        preprocess_pipeline,
        species_list,
        amr_antibiotics=amr_antibiotics,
        amr_year=None
    )
