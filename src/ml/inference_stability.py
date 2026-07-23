"""
Inference pipeline for stability (log Kf) prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from src.ml.inference_ligand_strength import predict_ligand_strength

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "saved"

def load_reference_data():
    metals = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "metal_properties.csv")
    ligands = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "ligands_raw.csv")
    return metals, ligands

def compute_stability_features(metal_symbol, ligand_name, coordination_number):
    metals_df, ligands_df = load_reference_data()

    m = metals_df[metals_df["symbol"] == metal_symbol].iloc[0]
    l = ligands_df[ligands_df["ligand_name"] == ligand_name].iloc[0]

    ligand_strength = predict_ligand_strength(ligand_name)["strength_score"]

    return {
        "metal_atomic_radius": m["atomic_radius"],
        "metal_oxidation_state": m["oxidation_state"],
        "metal_hardness": m["hsab_hardness"],
        "metal_electronegativity": m["electronegativity"],
        "ligand_strength_score": ligand_strength,
        "coordination_number": coordination_number,
        "delta_o": l.get("known_delta_o", np.nan),
        "hsab_match_score": float(m["hsab_hardness"] * ligand_strength),
        "radius_mismatch": abs(m["atomic_radius"] - l["donor_radius_mean"])
    }

def predict_stability(metal_symbol, ligand_name, coordination_number):
    model = joblib.load(MODEL_DIR / "stability_model.pkl")
    features = compute_stability_features(metal_symbol, ligand_name, coordination_number)

    df = pd.DataFrame([features])
    log_kf = float(model.predict(df)[0])

    return {
        "metal": metal_symbol,
        "ligand": ligand_name,
        "coordination_number": coordination_number,
        "log_kf": log_kf,
        "features_used": features
    }

if __name__ == "__main__":
    print(predict_stability("Fe", "NH3", 6))
