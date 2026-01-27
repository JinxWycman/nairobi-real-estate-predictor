# app/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PropertyFeatures(BaseModel):
    location: str
    bedrooms: int
    size_sqm: float

@app.post("/predict")
def predict_price(features: PropertyFeatures):
    # Prediction logic
    return {"predicted_price": price}