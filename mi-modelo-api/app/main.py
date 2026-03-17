from fastapi import FastAPI
from app.schemas import PredictionInput
from app.model import predict

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok", "model_version": "1.0"}

@app.post("/predict")
def predict_endpoint(data: PredictionInput):
    result = predict(data)
    return {"prediction": result}