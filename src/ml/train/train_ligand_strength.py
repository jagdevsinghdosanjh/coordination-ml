"""
Train ligand strength prediction model.
Uses donor atom descriptors + ligand metadata to predict a continuous strength score.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "processed" / "ligand_feature_matrix.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"

def main():
    df = pd.read_csv(DATA)

    feature_cols = [
        "donor_electronegativity_mean",
        "donor_ionization_energy_mean",
        "donor_radius_mean",
        "donor_hardness_mean",
        "denticity",
        "donor_count",
        "ligand_charge_density_proxy",
        "known_delta_o"
    ]

    X = df[feature_cols]
    y = df["ligand_strength"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\n=== Ligand Strength Model Report ===")
    print("R²:", r2_score(y_test, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "ligand_strength_model.pkl")

    print("\nModel saved to:", MODEL_DIR)

if __name__ == "__main__":
    main()
