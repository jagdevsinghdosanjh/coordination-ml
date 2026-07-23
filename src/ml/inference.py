"""
src/ml/inference.py

Inference pipeline for Coordination‑ML.
Loads the trained geometry classifier and computes predictions
for new metal–ligand combinations using the same feature logic
as the training pipeline.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
CONFIGS = PROJECT_ROOT / "configs"

# ---------------------------------------------------------
# Load Config
# ---------------------------------------------------------

def load_config(config_name="features.yaml"):
    config_path = CONFIGS / config_name
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------
# Load Model + Encoder
# ---------------------------------------------------------

def load_model():
    model_path = MODEL_DIR / "geometry_model.pkl"
    encoder_path = MODEL_DIR / "geometry_label_encoder.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Label encoder not found: {encoder_path}")

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder

# ---------------------------------------------------------
# Load Reference Data (atomic + ligand)
# ---------------------------------------------------------

def load_reference_data():
    atomic = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "atomic_properties.csv")
    ligands = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "ligands_raw.csv")
    return atomic, ligands

# ---------------------------------------------------------
# Compute ligand descriptors (same logic as training)
# ---------------------------------------------------------

def compute_ligand_features(ligand_name, ligands_df, atomic_df):
    if ligand_name not in ligands_df["ligand_name"].values:
        raise ValueError(f"Ligand '{ligand_name}' not found in ligands_raw.csv")

    row = ligands_df[ligands_df["ligand_name"] == ligand_name].iloc[0]

    donor_atoms = [a.strip() for a in str(row["donor_atoms"]).split(" ")]

    atomic_map = atomic_df.set_index("symbol")

    def agg(feature):
        vals = []
        for atom in donor_atoms:
            if atom in atomic_map.index:
                vals.append(atomic_map.loc[atom, feature])
        return float(np.mean(vals)) if vals else np.nan

    features = {
        "donor_electronegativity_mean": agg("electronegativity"),
        "donor_ionization_energy_mean": agg("ionization_energy"),
        "donor_radius_mean": agg("atomic_radius"),
        "donor_hardness_mean": agg("hsab_hardness"),
        "denticity": int(row["denticity"]),
        "donor_count": len(donor_atoms),
        "ligand_charge_density_proxy": int(row["denticity"]) / agg("atomic_radius"),
        "known_delta_o": row.get("known_delta_o", np.nan),
        "known_log_kf": row.get("known_log_kf", np.nan),
    }

    return features

# ---------------------------------------------------------
# Build feature row for inference
# ---------------------------------------------------------

def build_feature_row(metal_symbol, ligand_name, coordination_number):
    atomic_df, ligands_df = load_reference_data()
    ligand_features = compute_ligand_features(ligand_name, ligands_df, atomic_df)

    row = {
        "complex_name": f"{metal_symbol}_{ligand_name}_{coordination_number}",
        "metal_symbol": metal_symbol,
        "ligand_name": ligand_name,
        "coordination_number": coordination_number,
    }

    row.update(ligand_features)
    return pd.DataFrame([row])

# ---------------------------------------------------------
# Predict geometry
# ---------------------------------------------------------

def predict_geometry(metal_symbol, ligand_name, coordination_number):
    model, encoder = load_model()
    config = load_config()

    # Build feature row
    df = build_feature_row(metal_symbol, ligand_name, coordination_number)

    # Select only feature columns used during training
    feature_cols = config["feature_columns"]
    feature_cols = [c for c in feature_cols if c != "geometry_label"]  # exclude target

    X = df[feature_cols]

    # Predict
    y_pred = model.predict(X)
    label = encoder.inverse_transform(y_pred)[0]

    # Confidence score
    proba = model.predict_proba(X)[0]
    confidence = float(np.max(proba))

    return {
        "metal": metal_symbol,
        "ligand": ligand_name,
        "coordination_number": coordination_number,
        "predicted_geometry": label,
        "confidence": confidence,
        "probabilities": {
            encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(proba)
        }
    }

# ---------------------------------------------------------
# Manual test
# ---------------------------------------------------------

if __name__ == "__main__":
    result = predict_geometry("Fe", "NH3", 6)
    print(result)
