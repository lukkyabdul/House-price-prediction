from contextlib import asynccontextmanager
import os
import time
from typing import List
from fastapi import FastAPI, HTTPException, Request
import joblib
import numpy as np
from pydantic import BaseModel, Field, model_validator

MODEL_PATH = os.path.join(os.path.dirname(__file__), "house_price_model.joblib")
ml_models = {}


# 1. Lifespan Context Manager: Load ML model once at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run 'python src/train_model.py' first."
        )
    ml_models["house_price"] = joblib.load(MODEL_PATH)
    yield
    ml_models.clear()


app = FastAPI(
    title="House Price Prediction REST API",
    description="Team 09 Seminar Demo: Serving a Scikit-Learn Random Forest model via FastAPI.",
    version="1.0.0",
    lifespan=lifespan,
)


# 2. Custom Modification 3: Inference Latency Telemetry Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
    return response


# 3. Custom Modification 1: Pydantic Schema with Strict Domain Guardrails
class HouseFeatures(BaseModel):
    area_sqft: float = Field(
        ...,
        gt=250,
        le=20000,
        description="Total built-up area in sq.ft (must be > 250)",
    )
    bedrooms: int = Field(
        ..., ge=1, le=10, description="Number of bedrooms (1–10)"
    )
    bathrooms: int = Field(
        ..., ge=1, le=10, description="Number of bathrooms (1–10)"
    )
    age_years: int = Field(
        ..., ge=0, le=100, description="Property age in years (0–100)"
    )

    @model_validator(mode="after")
    def validate_room_density(self):
        # Prevent unrealistic inputs like 6 bedrooms inside 300 sq.ft
        min_required_sqft = self.bedrooms * 150
        if self.area_sqft < min_required_sqft:
            raise ValueError(
                f"area_sqft ({self.area_sqft}) is too small for {self.bedrooms} bedrooms "
                f"(minimum {min_required_sqft} sq.ft required)."
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "area_sqft": 1850,
                    "bedrooms": 3,
                    "bathrooms": 2,
                    "age_years": 5,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    predicted_price_lakhs: float
    currency: str = "INR"
    model_version: str = "RandomForest-v1"


class BatchPredictionRequest(BaseModel):
    houses: List[HouseFeatures] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions_lakhs: List[float]
    currency: str = "INR"


# 4. REST API Endpoints
@app.get("/health", tags=["Monitoring"])
def health_check():
    """Verify API server and ML model readiness."""
    model_loaded = "house_price" in ml_models
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_price(features: HouseFeatures):
    """Predict house price for a single property."""
    model = ml_models.get("house_price")
    if model is None:
        raise HTTPException(status_code=503, detail="ML model is not loaded.")

    input_array = np.array([[
        features.area_sqft,
        features.bedrooms,
        features.bathrooms,
        features.age_years,
    ]])
    prediction = float(model.predict(input_array)[0])

    return PredictionResponse(predicted_price_lakhs=round(prediction, 2))


# 5. Custom Modification 2: Vectorized Batch Prediction Endpoint
@app.post(
    "/predict-batch", response_model=BatchPredictionResponse, tags=["Inference"]
)
def predict_batch(payload: BatchPredictionRequest):
    """Predict house prices for multiple properties in a single vectorized call."""
    model = ml_models.get("house_price")
    if model is None:
        raise HTTPException(status_code=503, detail="ML model is not loaded.")

    batch_array = np.array([
        [h.area_sqft, h.bedrooms, h.bathrooms, h.age_years]
        for h in payload.houses
    ])
    predictions = [round(float(p), 2) for p in model.predict(batch_array)]

    return BatchPredictionResponse(
        count=len(predictions),
        predictions_lakhs=predictions,
    )