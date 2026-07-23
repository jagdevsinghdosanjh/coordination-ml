# api/routes/predict_ligand_strength.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.ml.inference_ligand_strength import predict_ligand_strength

router = APIRouter(
    prefix="/predict-ligand-strength",
    tags=["Ligand Strength Prediction"]
)

# -----------------------------
# Request Model
# -----------------------------
class LigandStrengthRequest(BaseModel):
    ligand_name: str

# -----------------------------
# Response Model
# -----------------------------
class LigandStrengthResponse(BaseModel):
    ligand: str
    strength_score: float
    features_used: dict

# -----------------------------
# Route
# -----------------------------
@router.post("/", response_model=LigandStrengthResponse)
def predict_ligand_strength_route(payload: LigandStrengthRequest):
    try:
        result = predict_ligand_strength(
            ligand_name=payload.ligand_name.strip()
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
