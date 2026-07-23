from fastapi import FastAPI
from api.routes.predict_geometry import router as geometry_router
from api.routes.predict_ligand_strength import router as ligand_strength_router
from api.routes.predict_stability import router as stability_router


# -----------------------------
# Create FastAPI App
# -----------------------------
app = FastAPI(
    title="Coordination‑ML API",
    version="1.0"
)

# -----------------------------
# Include Routers
# -----------------------------
app.include_router(geometry_router)
app.include_router(ligand_strength_router)
app.include_router(stability_router)