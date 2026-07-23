"""
features/build_feature_matrix.py

Builds the full ML feature matrix for the Coordination-ML project:
- Loads raw ligand / atomic / complex data
- Computes atomic, ligand, chelation, and geometry descriptors
- Saves processed feature matrix to data/processed/
"""

import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
import yaml


# -------------------------------------------------------------------
# Paths & Config Loading
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CONFIGS = PROJECT_ROOT / "configs"


def load_config(config_name: str = "features.yaml") -> Dict[str, Any]:
    config_path = CONFIGS / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------------------------------------------------
# Core loaders
# -------------------------------------------------------------------

def load_raw_data() -> Dict[str, pd.DataFrame]:
    ligands_path = DATA_RAW / "ligands_raw.csv"
    atomic_path = DATA_RAW / "atomic_properties.csv"
    complexes_path = DATA_RAW / "complexes_raw.csv"

    if not ligands_path.exists():
        raise FileNotFoundError(f"Missing ligands_raw.csv at {ligands_path}")
    if not atomic_path.exists():
        raise FileNotFoundError(f"Missing atomic_properties.csv at {atomic_path}")
    if not complexes_path.exists():
        raise FileNotFoundError(f"Missing complexes_raw.csv at {complexes_path}")

    ligands = pd.read_csv(ligands_path)
    atomic = pd.read_csv(atomic_path)
    complexes = pd.read_csv(complexes_path)

    return {
        "ligands": ligands,
        "atomic": atomic,
        "complexes": complexes,
    }


# -------------------------------------------------------------------
# Atomic descriptors
# -------------------------------------------------------------------

def compute_atomic_descriptors(atomic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect atomic_df to contain at least:
    - symbol
    - electronegativity
    - ionization_energy
    - atomic_radius
    - hsab_hardness
    """
    required_cols = [
        "symbol",
        "electronegativity",
        "ionization_energy",
        "atomic_radius",
        "hsab_hardness",
    ]
    missing = [c for c in required_cols if c not in atomic_df.columns]
    if missing:
        raise ValueError(f"Atomic properties missing columns: {missing}")

    # Example engineered feature: charge density proxy (neutral atoms → q=0, but keep structure)
    atomic_df["charge_density_proxy"] = 0.0

    return atomic_df


# -------------------------------------------------------------------
# Ligand descriptors
# -------------------------------------------------------------------

def compute_ligand_descriptors(
    ligands_df: pd.DataFrame,
    atomic_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Expect ligands_df to contain:
    - ligand_name
    - donor_atoms (comma-separated symbols, e.g. 'N,O')
    - denticity
    Optionally:
    - known_delta_o
    - known_log_kf
    """

    required_cols = ["ligand_name", "donor_atoms", "denticity"]
    missing = [c for c in required_cols if c not in ligands_df.columns]
    if missing:
        raise ValueError(f"Ligand data missing columns: {missing}")

    # Map atomic properties to donor atoms
    atomic_map = atomic_df.set_index("symbol")

    def parse_donor_atoms(row):
        atoms = [a.strip() for a in str(row["donor_atoms"]).split(",") if a.strip()]
        return atoms

    ligands_df["donor_atom_list"] = ligands_df.apply(parse_donor_atoms, axis=1)

    # Aggregate atomic descriptors over donor atoms
    def aggregate_atomic_feature(row, feature: str, agg: str = "mean"):
        atoms = row["donor_atom_list"]
        values = []
        for a in atoms:
            if a in atomic_map.index:
                values.append(atomic_map.loc[a, feature])
        if not values:
            return np.nan
        if agg == "mean":
            return float(np.mean(values))
        if agg == "max":
            return float(np.max(values))
        if agg == "min":
            return float(np.min(values))
        return float(np.mean(values))

    ligands_df["donor_electronegativity_mean"] = ligands_df.apply(
        aggregate_atomic_feature, axis=1, feature="electronegativity", agg="mean"
    )
    ligands_df["donor_ionization_energy_mean"] = ligands_df.apply(
        aggregate_atomic_feature, axis=1, feature="ionization_energy", agg="mean"
    )
    ligands_df["donor_radius_mean"] = ligands_df.apply(
        aggregate_atomic_feature, axis=1, feature="atomic_radius", agg="mean"
    )
    ligands_df["donor_hardness_mean"] = ligands_df.apply(
        aggregate_atomic_feature, axis=1, feature="hsab_hardness", agg="mean"
    )

    # Simple engineered features
    ligands_df["denticity"] = ligands_df["denticity"].astype(int)
    ligands_df["donor_count"] = ligands_df["donor_atom_list"].apply(len)

    # Charge density proxy: denticity / donor_radius_mean
    ligands_df["ligand_charge_density_proxy"] = (
        ligands_df["denticity"] / ligands_df["donor_radius_mean"]
    )

    # Ligand strength label (weak/medium/strong) from config thresholds
    strength_cfg = config.get("ligand_strength", {})
    delta_o_col = strength_cfg.get("delta_o_column", "known_delta_o")

    def strength_label(row):
        val = row.get(delta_o_col, np.nan)
        if pd.isna(val):
            return "unknown"
        if val < strength_cfg.get("weak_max", 150):
            return "weak"
        if val < strength_cfg.get("medium_max", 300):
            return "medium"
        return "strong"

    if delta_o_col in ligands_df.columns:
        ligands_df["strength_label"] = ligands_df.apply(strength_label, axis=1)
    else:
        ligands_df["strength_label"] = "unknown"

    return ligands_df


# -------------------------------------------------------------------
# Complex / geometry descriptors
# -------------------------------------------------------------------

def compute_complex_descriptors(
    complexes_df: pd.DataFrame,
    ligands_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Expect complexes_df to contain:
    - complex_name
    - metal_symbol
    - oxidation_state
    - ligand_name
    - coordination_number
    Optionally:
    - known_geometry
    - known_delta_o
    - known_log_kf
    """

    required_cols = [
        "complex_name",
        "metal_symbol",
        "oxidation_state",
        "ligand_name",
        "coordination_number",
    ]
    missing = [c for c in required_cols if c not in complexes_df.columns]
    if missing:
        raise ValueError(f"Complex data missing columns: {missing}")

    # Join ligand features
    ligands_features = ligands_df.set_index("ligand_name")
    complexes_df = complexes_df.join(
        ligands_features,
        on="ligand_name",
        rsuffix="_ligand",
    )

    # Simple geometry heuristic (can be replaced by ML later)
    def geometry_guess(row):
        cn = row["coordination_number"]
        if cn == 4:
            # square planar vs tetrahedral → use metal + ligand strength
            metal = str(row["metal_symbol"]).strip()
            strength = row.get("strength_label", "unknown")
            if metal in ["Ni", "Pd", "Pt"] and strength in ["strong", "medium"]:
                return "square_planar"
            return "tetrahedral"
        if cn == 6:
            return "octahedral"
        return "unknown"

    complexes_df["geometry_label"] = complexes_df.apply(geometry_guess, axis=1)

    return complexes_df


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def build_feature_matrix(config_name: str = "features.yaml") -> pd.DataFrame:
    config = load_config(config_name)
    raw = load_raw_data()

    atomic_df = compute_atomic_descriptors(raw["atomic"])
    ligands_df = compute_ligand_descriptors(raw["ligands"], atomic_df, config)
    complexes_df = compute_complex_descriptors(raw["complexes"], ligands_df, config)

    # Final feature matrix: one row per complex
    feature_cols = config.get("feature_columns", None)
    if feature_cols is None:
        # Default: all numeric columns + key identifiers
        id_cols = ["complex_name", "metal_symbol", "ligand_name"]
        numeric_cols = complexes_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = id_cols + numeric_cols

    feature_matrix = complexes_df[feature_cols].copy()

    # Ensure processed directory exists
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    out_csv = DATA_PROCESSED / "feature_matrix.csv"
    out_parquet = DATA_PROCESSED / "feature_matrix.parquet"

    feature_matrix.to_csv(out_csv, index=False)
    feature_matrix.to_parquet(out_parquet, index=False)

    print(f"✅ Feature matrix saved to:\n  - {out_csv}\n  - {out_parquet}")
    return feature_matrix


if __name__ == "__main__":
    build_feature_matrix()
