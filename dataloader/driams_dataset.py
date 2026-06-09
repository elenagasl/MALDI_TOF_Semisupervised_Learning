"""
DRIAMS Dataset Manager
=======================

Provides data loading and indexing for the DRIAMS (Drug Resistance In MALDI-TOF
Mass Spectrometry) multi-centre dataset (Weis et al., Nature Medicine, 2022).

Dataset structure on disk
--------------------------
::

    <DRIAMS_ROOT>/
    ├── DRIAMS-A/
    │   ├── id/
    │   │   ├── 2015/  2015_clean.csv   (metadata: code, species, AST results)
    │   │   └── 2016/  2016_clean.csv
    │   └── raw/
    │       ├── 2015/  <code>.txt       (tab-separated m/z, intensity)
    │       └── 2016/
    ├── DRIAMS-B/
    │   └── ...
    └── DRIAMS-C/

Classes
-------
DRIAMS_Manager
    Builds and queries an in-memory index of available spectra.
    Supports saving/loading the index to/from pickle for faster
    repeated initialisation.

    ``spectra_dict[center][year][species][code]`` → list of raw .txt paths

DRIAMS(Dataset)
    PyTorch Dataset wrapping the manager. Loads, preprocesses and
    returns (SpectrumObject, label, meta) tuples on demand.
"""

import os
import pickle
import random
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset

from dataloader.spectrum_object import SpectrumObject
from visualization.visualize import visualize_preprocessing

class DRIAMS_Manager:
    def __init__(self, root_dir, presaved=False, pickle_path=None):
        """
        Initializes the DRIAMSManager.

        spectra_dict structure:
            spectra_dict[site][year][species][code] -> list of raw .txt paths
        """
        self.root_dir = root_dir

        if not presaved:
            self.spectra_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
            self._load_structure()
        else:
            if pickle_path is None:
                raise ValueError("Pickle path must be provided if presaved is True.")
            if not os.path.exists(pickle_path):
                raise FileNotFoundError(f"Pickle file {pickle_path} does not exist.")
            
            # Load the spectra_dict from the pickle file
            self.spectra_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
            self.load_from_pickle(pickle_path)
        
        self.stats = self.compute_statistics()  # Precompute statistics

    def _load_structure(self):
        """
        Traverse the DRIAMS directory structure and build an index of available
        raw MALDI-TOF spectra using the metadata provided in the `id` folders.

        The DRIAMS dataset does not encode biological or clinical information
        in the folder hierarchy. Instead, species labels and sample identifiers
        are defined in the yearly metadata CSV files (`*_clean.csv`).

        This method reads the metadata files and constructs `spectra_dict` with
        the following structure:

            spectra_dict[center][year][species][code] -> list of raw spectrum paths

        where:
            - center: DRIAMS site identifier (e.g. "DRIAMS-A", "DRIAMS-B")
            - year: acquisition year as a string (e.g. "2015")
            - species: species label taken from the metadata CSV
            - code: unique spectrum identifier matching the raw `.txt` filename

        Only spectra for which the corresponding raw `.txt` file exists on disk
        are included in the index.
        """

        total_spectra = sum(
            1
            for center in os.listdir(self.root_dir) if center.startswith("DRIAMS_") if os.path.isdir(os.path.join(self.root_dir, center))
            for year in os.listdir(os.path.join(self.root_dir, center, "id")) if os.path.exists(os.path.join(self.root_dir, center, "id", year, f"{year}_clean.csv"))
            for _, row in pd.read_csv(os.path.join(self.root_dir, center, "id", year, f"{year}_clean.csv"), low_memory=False).iterrows() if not row["species"].startswith("MIX!") if os.path.exists(os.path.join(self.root_dir, center, "raw", year, f"{row['code']}.txt"))
        )
    
        print(f"Total spectra to process: {total_spectra}")

        with tqdm(total=total_spectra, desc="Indexing DRIAMS spectra") as pbar:
            for center in os.listdir(self.root_dir):
                center_path = os.path.join(self.root_dir, center)
                if not os.path.isdir(center_path):
                    continue

                id_path = os.path.join(center_path, "id")
                raw_path = os.path.join(center_path, "raw")
            
                for year in os.listdir(id_path):
                    year_csv = os.path.join(os.path.join(id_path, year), f"{year}_clean.csv")
                    year_raw = os.path.join(raw_path, year)

                    if not os.path.exists(year_csv):
                        continue

                    df = pd.read_csv(year_csv, low_memory=False)

                    for _, row in df.iterrows():
                        code = row["code"]
                        species = row["species"]
                        if species.startswith("MIX!"):
                            continue

                        txt_path = os.path.join(year_raw, f"{code}.txt")
                        if not os.path.exists(txt_path):
                            continue

                        genus, species = species.split(" ", 1)
                        genus = genus.capitalize()
                        species = species.lower()
                        species = species.replace(" ", "_")  
                        name = f"{genus}_{species.capitalize() if '_' not in species else species}"

                        self.spectra_dict[center][year][name][code].append(txt_path)
                        pbar.update(1)
    
    def compute_statistics(self):
        """
        Computes a statistics DataFrame for DRIAMS.

        Rows:
            - Species (Genus_Species)

        Columns:
            - (Year, 'Total'): total number of spectra
            - (Year, 'Unique'): number of unique samples (codes)

        Note:
            DRIAMS does not provide an isolate identifier.
            Each spectrum is associated with a unique code and is treated as an
            independent sample; therefore, Total == Unique.
        """
        # Diccionario auxiliar
        stats = {}

        # Recorrer toda la estructura
        for _, year_dict in self.spectra_dict.items():
            for year, species_dict in year_dict.items():
                for species_name, txt_dict in species_dict.items():
                    if species_name not in stats:
                        stats[species_name] = {}  
                        
                    total_count = len(txt_dict)

                    total_count = sum(len(v) for v in  txt_dict.values())
                    unique_count = len(txt_dict)  # number of codes

                    stats[species_name][(year, "Total")] = (
                        stats[species_name].get((year, "Total"), 0) + total_count
                    )
                    stats[species_name][(year, "Unique")] = (
                        stats[species_name].get((year, "Unique"), 0) + unique_count
                    )

        # Crear DataFrame
        self.stats = pd.DataFrame.from_dict(stats, orient='index')

        # Ordenar columnas: primero por año y luego Unique/Total
        self.stats = self.stats.sort_index(axis=1, level=[0, 1])

        # Añadir nombre del índice
        self.stats.index.name = 'Species'

        return self.stats
    
    def save_to_pickle(self, file_path):
        """
        Saves the manager object (including spectra_dict and stats) to a pickle file.
        """
        # Convertir defaultdict a dict normal (sin lambdas)
        spectra_dict_clean = self._defaultdict_to_dict(self.spectra_dict)

        with open(file_path, 'wb') as f:
            pickle.dump({'spectra_dict': spectra_dict_clean}, f)

        print(f"✅ Manager saved to {file_path}")

    def _defaultdict_to_dict(self, d):
        """
        Recursively convert defaultdicts into normal dicts.
        """
        if isinstance(d, defaultdict):
            d = {k: self._defaultdict_to_dict(v) for k, v in d.items()}
        return d

    def load_from_pickle(self, file_path):
        """
        Loads the manager object (spectra_dict and stats) from a pickle file.
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.spectra_dict = data['spectra_dict']
        print(f"✅ Manager loaded from {file_path}")

    def query_spectra_dict(self, centers=None, years=None, genus_species=None):
        """
        Query the DRIAMS spectra_dict with optional filters.

        Parameters
        ----------
        centers : str or list of str or None
            DRIAMS centers (e.g. "DRIAMS_A")
        years : str or list of str or None
            Acquisition years (e.g. "2018")
        genus_species : str or list of str or None
            Species names exactly as stored (e.g. "Klebsiella_Pneumoniae")

        Returns
        -------
        filtered_dict : defaultdict
            Filtered spectra_dict with the same nested structure:
            filtered[center][year][genus_species][code] -> list of paths
        """
        # Normalize inputs to lists
        if isinstance(centers, str):
            centers = [centers]
        if isinstance(years, str):
            years = [years]
        if genus_species is not None:
            genus_species = {
                f"{g}_{s}"
                for g, s in genus_species
            }

        if isinstance(genus_species, str):
            genus_species = [genus_species]

        filtered = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

        for center, year_dict in self.spectra_dict.items():
            if centers and center not in centers:
                continue

            for year, species_dict in year_dict.items():
                if years and year not in years:
                    continue

                for gs_name, code_dict in species_dict.items():
                    if genus_species and gs_name not in genus_species:
                        continue

                    for code, paths in code_dict.items():
                        filtered[center][year][gs_name][code].extend(paths)

        return filtered
    
    def get_top_species(self, top_n=5):
        total_col = self.stats.columns.get_level_values(1) == "Total"
        totals = self.stats.loc[:, total_col].sum(axis=1)

        top_species = (
            totals
            .sort_values(ascending=False)
            .head(top_n)
            .index
            .tolist()
        )

        return [tuple(s.split('_', 1)) for s in top_species]


class DRIAMS(Dataset):
    def __init__(self, spectra_dict, preprocess_pipeline=None, visualize=False, path=None):
        self.samples = []
        self.preprocess_pipeline = preprocess_pipeline
        self.spectra_dict = spectra_dict

        for center, year_dict in spectra_dict.items():
            for year, species_dict in year_dict.items():
                for genus_species, codes in species_dict.items():
                    genus, species = genus_species.split("_", 1)

                    for code, txt_paths in codes.items():
                        for txt_path in txt_paths:
                            self.samples.append({
                                'path': txt_path,
                                'label': genus_species,  
                                'meta': {
                                    'hospital': center,
                                    'year': year,
                                    'genus': genus,
                                    'species': species,
                                    'study': code
                                }
                            })

        if visualize:
            if path is None:
                raise ValueError("Path must be provided for visualization.")
            if len(self.samples) > 0:
                sample = random.choice(self.samples)
                spectrum = SpectrumObject.from_tsv(sample["path"])
                visualize_preprocessing(
                    (spectrum, sample["meta"]),
                    preprocess_pipeline,
                    path
                )
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]

        # Load and preprocess spectrum
        spectrum = SpectrumObject.from_tsv(entry["path"])
        if self.preprocess_pipeline:
            spectrum = self.preprocess_pipeline(spectrum)

        # Return SpectrumObject and label
        return SpectrumObject(mz=spectrum.mz, intensity=spectrum.intensity), entry['label'], entry['meta']
