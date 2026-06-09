#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
from pathlib import Path


MSUMG_PATH = "/export/data_ml4ds/bacteria_id/MALDIAlign_Alex/MSUMG_study_full.pkl"
OUTPUT_DIR = Path("./msumg_inspection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_label(x):
    return str(x).replace(" ", "_")


def main():
    print(f"Loading: {MSUMG_PATH}")

    with open(MSUMG_PATH, "rb") as f:
        payload = pickle.load(f)

    print("\n==============================")
    print("PKL KEYS")
    print("==============================")
    print(payload.keys())

    data = np.asarray(payload["data"])
    labels = np.asarray(payload["label"]).astype(str)
    meta = pd.DataFrame(payload["meta"])

    print("\n==============================")
    print("BASIC SHAPES")
    print("==============================")
    print("data shape:", data.shape)
    print("label shape:", labels.shape)
    print("meta shape:", meta.shape)

    print("\n==============================")
    print("META COLUMNS")
    print("==============================")
    print(meta.columns.tolist())

    if "agar" in meta.columns:
        print("\n==============================")
        print("AGAR DISTRIBUTION")
        print("==============================")
        print(meta["agar"].value_counts(dropna=False))

    if "year" in meta.columns:
        print("\n==============================")
        print("YEAR DISTRIBUTION")
        print("==============================")
        print(meta["year"].value_counts(dropna=False).sort_index())

    species_counts = pd.Series(labels).value_counts().reset_index()
    species_counts.columns = ["species", "n_samples"]

    print("\n==============================")
    print("TOP SPECIES")
    print("==============================")
    print(species_counts.head(50).to_string(index=False))

    species_counts.to_csv(OUTPUT_DIR / "msumg_species_counts.csv", index=False)

    has_amr = "amr" in payload and "antibiotics" in payload

    print("\n==============================")
    print("HAS AMR?")
    print("==============================")
    print(has_amr)

    if has_amr:
        amr = np.asarray(payload["amr"], dtype=float)
        antibiotics = [str(a) for a in payload["antibiotics"]]

        print("amr shape:", amr.shape)
        print("n antibiotics:", len(antibiotics))
        print("antibiotics:")
        for ab in antibiotics:
            print(" -", ab)

        ab_rows = []

        for j, ab in enumerate(antibiotics):
            col = amr[:, j]
            valid = np.isfinite(col)
            n_obs = int(valid.sum())

            if n_obs > 0:
                n_s = int(np.sum(col[valid] == 0))
                n_r = int(np.sum(col[valid] == 1))
                n_other = int(n_obs - n_s - n_r)
            else:
                n_s = 0
                n_r = 0
                n_other = 0

            ab_rows.append({
                "antibiotic": ab,
                "n_observed": n_obs,
                "n_susceptible_0": n_s,
                "n_resistant_1": n_r,
                "n_other": n_other,
                "resistance_rate": n_r / n_obs if n_obs > 0 else np.nan,
            })

        ab_df = pd.DataFrame(ab_rows).sort_values("n_observed", ascending=False)
        ab_df.to_csv(OUTPUT_DIR / "msumg_antibiotic_counts.csv", index=False)

        print("\n==============================")
        print("ANTIBIOTIC COUNTS")
        print("==============================")
        print(ab_df.to_string(index=False))

        rows = []

        for sp in sorted(np.unique(labels)):
            sp_mask = labels == sp
            amr_sp = amr[sp_mask]

            for j, ab in enumerate(antibiotics):
                col = amr_sp[:, j]
                valid = np.isfinite(col)
                n_obs = int(valid.sum())

                if n_obs == 0:
                    continue

                n_s = int(np.sum(col[valid] == 0))
                n_r = int(np.sum(col[valid] == 1))

                rows.append({
                    "species": sp,
                    "antibiotic": ab,
                    "n_samples_species": int(sp_mask.sum()),
                    "n_observed": n_obs,
                    "n_susceptible_0": n_s,
                    "n_resistant_1": n_r,
                    "has_two_classes": int(n_s > 0 and n_r > 0),
                    "resistance_rate": n_r / n_obs if n_obs > 0 else np.nan,
                })

        sp_ab_df = pd.DataFrame(rows)
        sp_ab_df.to_csv(OUTPUT_DIR / "msumg_species_antibiotic_counts.csv", index=False)

        print("\n==============================")
        print("TOP SPECIES-ANTIBIOTIC OBSERVED PAIRS")
        print("==============================")
        print(
            sp_ab_df
            .sort_values("n_observed", ascending=False)
            .head(100)
            .to_string(index=False)
        )

        eligible = sp_ab_df[
            (sp_ab_df["n_observed"] >= 20) &
            (sp_ab_df["has_two_classes"] == 1)
        ].copy()

        eligible.to_csv(OUTPUT_DIR / "msumg_species_antibiotic_eligible_min20.csv", index=False)

        print("\n==============================")
        print("ELIGIBLE SPECIES-ANTIBIOTIC PAIRS, MIN 20 OBSERVED AND TWO CLASSES")
        print("==============================")
        print(eligible.to_string(index=False))

    print("\nSaved outputs in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()