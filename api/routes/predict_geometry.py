# api/routes/predict_geometry.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.ml.inference import predict_geometry

router = APIRouter(
    prefix="/predict-geometry",
    tags=["Geometry Prediction"]
)

# -----------------------------
# Request Model
# -----------------------------
class GeometryRequest(BaseModel):
    metal_symbol: str
    ligand_name: str
    coordination_number: int

# -----------------------------
# Response Model
# -----------------------------
class GeometryResponse(BaseModel):
    metal: str
    ligand: str
    coordination_number: int
    predicted_geometry: str
    confidence: float
    probabilities: dict

# -----------------------------
# Route
# -----------------------------
@router.post("/", response_model=GeometryResponse)
def predict_geometry_route(payload: GeometryRequest):
    try:
        result = predict_geometry(
            metal_symbol=payload.metal_symbol.strip(),
            ligand_name=payload.ligand_name.strip(),
            coordination_number=payload.coordination_number
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
