"""A FastAPI service that scores a molecule for activity against the glucocorticoid receptor.

Send a SMILES string, get back the descriptors the model saw, the probability it is active,
and the call. The model is loaded once when the process starts, not per request.
"""
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from train_model import descriptors

MODEL_PATH = Path(__file__).parent / "model.joblib"
THRESHOLD = 0.5

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]

app = FastAPI(
    title="Bioactivity API",
    description="Predicts whether a compound is active against the glucocorticoid receptor, "
                "from its SMILES string.",
    version="1.0.0",
)


class Molecule(BaseModel):
    smiles: str = Field(..., examples=["CC(=O)Oc1ccccc1C(=O)O"], description="SMILES string")


class Prediction(BaseModel):
    smiles: str
    active: bool
    probability: float
    descriptors: dict


@app.get("/health")
def health() -> dict:
    """Used by a container health check: is the process up and is the model loaded?"""
    return {"status": "ok", "model_loaded": model is not None, "features": len(FEATURES)}


@app.post("/predict", response_model=Prediction)
def predict(molecule: Molecule) -> Prediction:
    values = descriptors(molecule.smiles)
    if values is None:
        raise HTTPException(status_code=422, detail=f"RDKit could not parse SMILES: {molecule.smiles}")

    row = np.array([[values[name] for name in FEATURES]], dtype=float)
    probability = float(model.predict_proba(row)[0, 1])
    return Prediction(
        smiles=molecule.smiles,
        active=probability >= THRESHOLD,
        probability=round(probability, 3),
        descriptors={k: round(float(v), 3) for k, v in values.items()},
    )
