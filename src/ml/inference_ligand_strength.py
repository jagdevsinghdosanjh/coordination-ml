"""
Inference pipeline for ligand strength prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "saved"

def load_reference_data():
    atomic = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "atomic_properties.csv")
    ligands = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "ligands_raw.csv")
    return atomic, ligands

def compute_ligand_features(ligand_name):
    atomic_df, ligands_df = load_reference_data()

    row = ligands_df[ligands_df["ligand_name"] == ligand_name].iloc[0]
    donor_atoms = [a.strip() for a in row["donor_atoms"].split(" ")]

    atomic_map = atomic_df.set_index("symbol")

    def agg(feature):
        vals = []
        for atom in donor_atoms:
            if atom in atomic_map.index:
                vals.append(atomic_map.loc[atom, feature])
        return float(np.mean(vals)) if vals else np.nan

    return {
        "donor_electronegativity_mean": agg("electronegativity"),
        "donor_ionization_energy_mean": agg("ionization_energy"),
        "donor_radius_mean": agg("atomic_radius"),
        "donor_hardness_mean": agg("hsab_hardness"),
        "denticity": int(row["denticity"]),
        "donor_count": len(donor_atoms),
        "ligand_charge_density_proxy": int(row["denticity"]) / agg("atomic_radius"),
        "known_delta_o": row.get("known_delta_o", np.nan)
    }

def predict_ligand_strength(ligand_name):
    model = joblib.load(MODEL_DIR / "ligand_strength_model.pkl")
    features = compute_ligand_features(ligand_name)

    df = pd.DataFrame([features])
    strength = float(model.predict(df)[0])

    return {
        "ligand": ligand_name,
        "strength_score": strength,
        "features_used": features
    }

if __name__ == "__main__":
    print(predict_ligand_strength("NH3"))
