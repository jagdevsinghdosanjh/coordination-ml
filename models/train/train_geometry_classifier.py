"""
Train a geometry classifier for coordination complexes.
Uses the feature matrix generated in data/processed/.
"""

import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"

def load_feature_matrix():
    path = DATA_PROCESSED / "feature_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {path}")
    return pd.read_csv(path)

def train_geometry_classifier():
    df = load_feature_matrix()

    # Target variable
    y = df["geometry_label"]

    # Drop non-numeric columns
    X = df.drop(columns=["complex_name", "metal_symbol", "ligand_name", "geometry_label"])

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print("\n=== Geometry Classifier Report ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model + label encoder
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "geometry_model.pkl")
    joblib.dump(le, MODEL_DIR / "geometry_label_encoder.pkl")

    print(f"\nModel saved to: {MODEL_DIR}")

if __name__ == "__main__":
    train_geometry_classifier()
