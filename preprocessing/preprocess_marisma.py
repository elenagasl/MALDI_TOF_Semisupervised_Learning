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

from dataloader.marisma_dataset import MARISMaManager, MARISMa
from preprocessing.pipeline import (
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
    amrs = []
    sample_types = []

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

    # ==========================
    # AMR EXTRACTION
    # ==========================
    amr_map = None
    antibiotic_cols = None
    sample_map = None

    if amr_antibiotics:
        amr_csv = os.path.join(dataset_path, "AMR.csv")

        if os.path.exists(amr_csv):
            amr_df = pd.read_csv(amr_csv, low_memory=False)

            # Create sample map
            sample_map = {}
            if "Sample" in amr_df.columns and "Path" in amr_df.columns:
                for _, row in amr_df.iterrows():
                    sample_map[row["Path"]] = row.get("Sample", None)
            else:
                logger.warning("No 'Sample' or 'Path' column found in AMR.csv")

            if amr_year is not None:
                try:
                    amr_df = amr_df[amr_df["Year"] == int(amr_year)]
                except Exception:
                    amr_df = amr_df[amr_df["Year"].astype(str) == str(amr_year)]

                logger.info(
                    f"Filtered AMR table to year {amr_year}: {len(amr_df)} rows"
                )

            antibiotic_cols = [ab for ab in amr_antibiotics if ab in amr_df.columns]

            if len(antibiotic_cols) == 0:
                logger.warning("No requested antibiotic columns found in AMR.csv")
            else:
                amr_df = amr_df.dropna(subset=antibiotic_cols, how="all")

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
                if "Path" not in amr_df.columns:
                    logger.warning("No 'Path' column found in AMR.csv; AMR extraction disabled")
                    amr_map = None
                else:
                    for _, row in amr_df.iterrows():
                        amr_map[row["Path"]] = encode_amr(row)

                    logger.info(
                        f"Created AMR map for {len(amr_map)} samples with AMR data"
                    )
        else:
            logger.warning(f"AMR.csv not found at {amr_csv}; AMR extraction disabled")

    # ==========================
    # PROCESS SPECTRA
    # ==========================
    logger.info("Processing species spectra...")

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

            sample_type = None
            if sample_map is not None:
                sample_type = sample_map.get(full_path, None)

            amrs.append(amr_vec)
            sample_types.append(sample_type)

        spectra.append(spectrum_obj.intensity)
        labels.append(label)
        metas.append(meta)

    data = np.stack(spectra) if len(spectra) > 0 else np.empty((0,))
    label = np.array(labels)
    meta = np.array(metas, dtype=object)

    amr_arr = np.stack(amrs) if amr_map is not None and len(amrs) > 0 else None
    sample_types_arr = (
        np.array(sample_types, dtype=object)
        if amr_map is not None and len(sample_types) > 0
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

    if amr_arr is not None:
        logger.info(f"Final AMR shape: {amr_arr.shape}")
    if sample_types_arr is not None:
        logger.info(f"Final sample_type shape: {sample_types_arr.shape}")

    return data, label, meta, amr_arr, antibiotic_cols, sample_types_arr


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
    os.makedirs(PICKLE_OUTPUT_DIR, exist_ok=True)

    logger.info("=" * 80)
    logger.info(
        f"PICKLE CREATION SESSION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info("=" * 80)
    logger.info(f"Pickle name: {name_pickle}")
    logger.info(f"Save path: {save_path}")
    logger.info(f"Dataset source: {dataset_path}")

    data, label, meta, amr_arr, antibiotic_cols, sample_types_arr = collect_species(
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

    payload = {
        "data": data,
        "label": label,
        "meta": meta,
        "hospital": 3,
    }

    if amr_arr is not None:
        payload["amr"] = amr_arr
        payload["antibiotics"] = antibiotic_cols

    if sample_types_arr is not None:
        payload["sample_type"] = sample_types_arr

    with open(save_path, "wb") as f:
        pickle.dump(payload, f)

    logger.info("Pickle saved successfully!")
    logger.info("=" * 80)
    logger.info("SESSION COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    name = "MARISMA_samples"

    species_list = [
        ("Staphylococcus", "Aureus"),
        ("Staphylococcus", "Epidermidis"),
        ("Escherichia", "Coli"),
        ("Klebsiella", "Pneumoniae"),
        ("Pseudomonas", "Aeruginosa"),
        ("Enterobacter", "Cloacae"),
        ("Proteus", "Mirabilis"),
        ("Staphylococcus", "Hominis"),
        ("Serratia", "Marcescens"),
        ("Staphylococcus", "Capitis"),
        ("Enterococcus", "Faecium"),
        ("Klebsiella", "Oxytoca"),
        ("Klebsiella", "Variicola"),
        ("Citrobacter", "Koseri"),
        ("Enterococcus", "Faecalis"),
        ("Staphylococcus", "Lugdunensis"),
        ("Citrobacter", "Freundii"),
        ("Morganella", "Morganii"),
        ("Proteus", "Vulgaris"),
        ("Staphylococcus", "Haemolyticus"),
        ("Candida", "Albicans"),
        ("Streptococcus", "Pneumoniae"),
        ("Stenotrophomonas", "Maltophilia"),
        ("Campylobacter", "Jejuni"),
        ("Haemophilus", "Influenzae"),
    ]

    amr_antibiotics = [
        "5-Fluorocytosine",
        "Amikacin",
        "Amoxicillin",
        "Amoxicillin-Clavulanic acid",
        "Amoxicillin-Clavulanic acid_uncomplicated_HWI",
        "Amphotericin B",
        "Ampicillin",
        "Ampicillin-Sulbactam",
        "Anidulafungin",
        "Azithromycin",
        "Aztreonam",
        "Bacitracin",
        "Benzylpenicillin",
        "Benzylpenicillin_others",
        "Benzylpenicillin_with_meningitis",
        "Benzylpenicillin_with_pneumonia",
        "Caspofungin",
        "Cefalotin-Cefazolin",
        "Cefazolin",
        "Cefepime",
        "Cefixime",
        "Cefotaxime",
        "Cefoxitin",
        "Cefoxitin_screen",
        "Cefpodoxime",
        "Ceftarolin",
        "Ceftazidime",
        "Ceftazidime-Avibactam",
        "Ceftobiprole",
        "Ceftolozane-Tazobactam",
        "Ceftriaxone",
        "Cefuroxime",
        "Cefuroxime.1",
        "Chloramphenicol",
        "Ciprofloxacin",
        "Clarithromycin",
        "Clindamycin",
        "Clindamycin_induced",
        "Colistin",
        "Cotrimoxazol",
        "Cotrimoxazole",
        "Daptomycin",
        "Doxycycline",
        "Ertapenem",
        "Erythromycin",
        "Ethambutol_5mg-l",
        "Fluconazole",
        "Fosfomycin",
        "Fusidic acid",
        "Gentamicin",
        "Gentamicin_high_level",
        "Imipenem",
        "Isavuconazole",
        "Isoniazid_.1mg-l",
        "Isoniazid_.4mg-l",
        "Itraconazole",
        "Levofloxacin",
        "Linezolid",
        "MRSA",
        "Meropenem",
        "Meropenem-Vaborbactam",
        "Meropenem_with_meningitis",
        "Meropenem_with_pneumonia",
        "Meropenem_without_meningitis",
        "Metronidazole",
        "Micafungin",
        "Minocycline",
        "Moxifloxacin",
        "Mupirocin",
        "Nitrofurantoin",
        "Norfloxacin",
        "Novobiocin",
        "Ofloxacin",
        "Oxacillin",
        "Pefloxacin",
        "Penicillin",
        "Penicillin_with_endokarditis",
        "Penicillin_with_meningitis",
        "Penicillin_with_other_infections",
        "Penicillin_with_pneumonia",
        "Penicillin_without_endokarditis",
        "Penicillin_without_meningitis",
        "Piperacillin",
        "Piperacillin-Tazobactam",
        "Polymyxin B",
        "Posaconazole",
        "Pristinamycin",
        "Pyrazinamide",
        "Rifampicin",
        "Rifampicin_1mg-l",
        "Sparfloxacin",
        "Strepomycin_high_level",
        "Streptomycin",
        "Teicoplanin",
        "Teicoplanin_GRD",
        "Telithromycin",
        "Tetracycline",
        "Ticarcillin",
        "Ticarcillin-Clavulan acid",
        "Tigecycline",
        "Tobramycin",
        "Vancomycin",
        "Vancomycin_GRD",
        "Voriconazole",
    ]

    amr_year = None
    change_names = {}

    preprocess_pipeline = SequentialPreprocessor(
        VarStabilizer(method="sqrt"),
        Smoother(halfwindow=10),
        BaselineCorrecter(method="SNIP", snip_n_iter=20),
        StdThresholder(factor=1.0),
        Trimmer(min=2000, max=20000),
        Binner(start=2000, stop=20000, step=3, aggregation="mean"),
        LogScaler(base=10),
    )

    main(
        name,
        preprocess_pipeline,
        species_list,
        change_names,
        amr_antibiotics=amr_antibiotics,
        amr_year=amr_year,
    )