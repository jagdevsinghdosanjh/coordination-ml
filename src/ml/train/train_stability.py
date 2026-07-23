"""
Train stability (log Kf) prediction model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "processed" / "stability_feature_matrix.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"

def main():
    df = pd.read_csv(DATA)

    feature_cols = [
        "metal_atomic_radius",
        "metal_oxidation_state",
        "metal_hardness",
        "metal_electronegativity",
        "ligand_strength_score",
        "coordination_number",
        "delta_o",
        "hsab_match_score",
        "radius_mismatch"
    ]

    X = df[feature_cols]
    y = df["log_kf"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=400, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\n=== Stability Model Report ===")
    print("R²:", r2_score(y_test, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "stability_model.pkl")

    print("\nModel saved to:", MODEL_DIR)

if __name__ == "__main__":
    main()
