from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.inference import Predictor

app = FastAPI(
    title="Battery Failure Predictor",
    description="Predicts EV battery failure risk from telemetry features.",
    version="1.0.0",
)

predictor = Predictor(artifacts_dir="artifacts")

app.mount("/static", StaticFiles(directory="app/static"), name="static")


class PredictRequest(BaseModel):
    data: Dict[str, Any]


@app.get("/")
def root():
    return FileResponse("app/static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def features():
    """
    Returns the exact feature list the model expects, plus the valid
    category options for each categorical feature (used to build the
    web form dynamically).
    """
    return {
        "num_cols": predictor.num_cols,
        "cat_cols": predictor.cat_cols,
        "categories": {
            col: predictor.label_encoders[col].classes_.tolist()
            for col in predictor.cat_cols
        },
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return predictor.predict(req.data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
