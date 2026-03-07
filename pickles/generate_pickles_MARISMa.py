import os
import sys
from tqdm import tqdm
import numpy as np
import pickle
import logging
from datetime import datetime
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from dataloader.MARISMa_Manager import MARISMaManager, MARISMa
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

from utils.config import MARISMA_ROOT, MARISMA_ANON_PICKLE, PICKLE_OUTPUT_DIR


def collect_species(
    dataset_path,
    preprocess_pipeline,
    species_list,
    logger,
    amr_antibiotics=None,
    amr_year=None,
):

    logger.info("Preprocessing pipeline:")
    for i, step in enumerate(preprocess_pipeline.preprocessors, 1):
        args = step.__dict__
        arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        logger.info(f"  {i}. {step.__class__.__name__}({arg_str})")

    # Load base anonymized pickle
    pickle_path = str(MARISMA_ANON_PICKLE)
    manager = MARISMaManager(dataset_path, presaved=True, pickle_path=pickle_path)

    spectra = []
    labels = []
    metas = []

    logger.info("################## PROCESSING SPECIES ##################:")
    logger.info("Bacteria species included:")
    for species in species_list:
        logger.info(f"  - {species[0]} {species[1].lower()}")

    logger.info("Querying species data...")
    species = (
        manager.query_spectra_dict(years=str(amr_year), genus_species=species_list)
        if amr_year is not None
        else manager.query_spectra_dict(genus_species=species_list)
    )

    logger.info("Creating dataset...")
    dataset = MARISMa(species, preprocess_pipeline=preprocess_pipeline)

    # AMR EXTRACTION

    amr_map = None
    antibiotic_cols = None

    if amr_antibiotics:
        amr_csv = os.path.join(dataset_path, "AMR.csv")

        if os.path.exists(amr_csv):
            amr_df = pd.read_csv(amr_csv, low_memory=False)

            if amr_year is not None:
                try:
                    amr_df = amr_df[amr_df["Year"] == int(amr_year)]
                except Exception:
                    amr_df = amr_df[amr_df["Year"].astype(str) == str(amr_year)]

                logger.info(
                    f"Filtered AMR table to year {amr_year}: {len(amr_df)} rows"
                )

            antibiotic_cols = [
                ab for ab in amr_antibiotics if ab in amr_df.columns
            ]

            if len(antibiotic_cols) == 0:
                logger.warning(
                    "No requested antibiotic columns found in AMR.csv"
                )
            else:
                amr_df = amr_df.dropna(
                    subset=antibiotic_cols, how="all"
                )

                def encode_amr(row):
                    vals = []
                    for abx in antibiotic_cols:
                        val = row.get(abx, None)
                        if pd.isna(val):
                            vals.append(np.nan)
                        else:
                            s = str(val).strip().upper()
                            if s in ["R", "I"]:
                                vals.append(1.0)
                            elif s == "S":
                                vals.append(0.0)
                            else:
                                vals.append(np.nan)
                    return np.array(vals, dtype=float)

                amr_map = {}

                for _, row in amr_df.iterrows():
                    amr_map[row["Path"]] = encode_amr(row)

                logger.info(
                    f"Created AMR map for {len(amr_map)} samples with AMR data"
                )

        else:
            logger.warning(
                f"AMR.csv not found at {amr_csv}; AMR extraction disabled"
            )

    # ==========================
    # PROCESS SPECTRA
    # ==========================

    logger.info("Processing species spectra...")
    amrs = []

    for i in tqdm(range(len(dataset)), desc="Extracting species"):

        spectrum_obj, label, meta = dataset[i]

        if amr_map is not None:
            study_base = meta.get("study", "")
            study_base = (
                study_base.rsplit("/", 1)[0]
                if "/" in study_base
                else study_base
            )

            full_path = (
                f"/MARISMa/{meta.get('year')}/"
                f"{meta.get('genus')}/"
                f"{meta.get('species')}/"
                f"{study_base}"
            )

            if full_path not in amr_map:
                continue

            amr_vec = amr_map[full_path]

            if np.all(np.isnan(amr_vec)):
                continue

            amrs.append(amr_vec)

        spectra.append(spectrum_obj.intensity)
        labels.append(label)
        metas.append(meta)

    data = np.stack(spectra) if len(spectra) > 0 else np.empty((0,))
    label = np.array(labels)
    meta = np.array(metas)

    amr_arr = (
        np.stack(amrs)
        if amr_map is not None and len(amrs) > 0
        else None
    )

    logger.info(f"Final data shape: {data.shape}")
    logger.info(f"Final label shape: {label.shape}")

    unique_classes = np.unique(label)
    logger.info(f"Unique classes ({len(unique_classes)}): {list(unique_classes)}")

    logger.info("Class distribution:")
    for class_name in unique_classes:
        count = np.sum(label == class_name)
        logger.info(f"  - {class_name}: {count} samples")

    return data, label, meta, amr_arr, antibiotic_cols


# MAIN

def main(
    name,
    preprocess_pipeline,
    species_list,
    change_names=None,
    amr_antibiotics=None,
    amr_year=None,
):

    log_file = os.path.join(
        os.path.dirname(__file__), f"pickle_creation_{name}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger("report_pickle_logger")

    dataset_path = str(MARISMA_ROOT)

    name_pickle = f"MARISMa_study_{name}"
    save_path = PICKLE_OUTPUT_DIR / f"{name_pickle}.pkl"

    logger.info("=" * 80)
    logger.info(
        f"PICKLE CREATION SESSION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info("=" * 80)
    logger.info(f"Pickle name: {name_pickle}")
    logger.info(f"Save path: {save_path}")
    logger.info(f"Dataset source: {dataset_path}")

    data, label, meta, amr_arr, antibiotic_cols = collect_species(
        dataset_path,
        preprocess_pipeline,
        species_list,
        logger,
        amr_antibiotics=amr_antibiotics,
        amr_year=amr_year,
    )

    if change_names:
        logger.info("Updating labels according to change_names mapping...")
        label = np.array([change_names.get(l, l) for l in label])

    logger.info(f"Saving pickle to: {save_path}")

    payload = {"data": data, "label": label, "meta": meta}

    if amr_arr is not None:
        payload["amr"] = amr_arr
        payload["antibiotics"] = antibiotic_cols

    with open(save_path, "wb") as f:
        pickle.dump(payload, f)

    logger.info("Pickle saved successfully!")
    logger.info("=" * 80)
    logger.info("SESSION COMPLETED")
    logger.info("=" * 80)

# EXECUTION BLOCK

if __name__ == "__main__":

    name = "MARISMA_half_pipeline"

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

    amr_year = None

    change_names = {}

    preprocess_pipeline = SequentialPreprocessor(
        #VarStabilizer(method="sqrt"),
        #Smoother(halfwindow=10),
        #BaselineCorrecter(method="SNIP", snip_n_iter=20),
        #StdThresholder(factor=1.0),
        Trimmer(min=2000, max=10000),
        Binner(start=2000, stop=10000, step=5, aggregation="mean"),
        #LogScaler(base=10),
    )

    main(
        name,
        preprocess_pipeline,
        species_list,
        change_names,
        amr_antibiotics=amr_antibiotics,
        amr_year=amr_year,
    )
