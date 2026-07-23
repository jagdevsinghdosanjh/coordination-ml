from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.ml.inference_stability import predict_stability

router = APIRouter(
    prefix="/predict-stability",
    tags=["Stability Prediction"]
)

class StabilityRequest(BaseModel):
    metal_symbol: str
    ligand_name: str
    coordination_number: int

class StabilityResponse(BaseModel):
    metal: str
    ligand: str
    coordination_number: int
    log_kf: float
    features_used: dict

@router.post("/", response_model=StabilityResponse)
def predict_stability_route(payload: StabilityRequest):
    try:
        result = predict_stability(
            metal_symbol=payload.metal_symbol.strip(),
            ligand_name=payload.ligand_name.strip(),
            coordination_number=payload.coordination_number
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
