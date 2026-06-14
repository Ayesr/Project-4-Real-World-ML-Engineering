 # PART 1: BUILD FASTAPI SERVICE
## Task 1.1: Create API Structure
import time
import logging
import pandas as pd
import skops.io as sio
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Annotated

# ============================================================
# 1. Configuration & Logging
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Path verified in your notebook
MODEL_PATH = r"C:\Users\Ayesha\mlartifacts\2\models\m-fce3960842d54c73a9cab93bb81abe25\artifacts\model.skops"
model = None

# ============================================================
# 2. Application Lifecycle
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        untrusted_types = sio.get_untrusted_types(file=MODEL_PATH)
        model = sio.load(MODEL_PATH, trusted=untrusted_types)
        logger.info("Successfully loaded model from local disk path")
    except Exception as e:
        logger.error(f"FATAL: Local model load failed: {str(e)}")
        model = None
    yield
    logger.info("Application shutdown")

# ============================================================
# 3. Initialize App (DEFINED HERE SO ROUTES CAN USE IT)
# ============================================================
app = FastAPI(
    title="Predictive Maintenance API",
    description="""
    ## Predictive Maintenance API
    This API uses machine learning to predict equipment failures based on real-time sensor data.
    
    ### Features:
    * **Failure Prediction:** Real-time analysis of sensor readings.
    * **Risk Assessment:** Categorized output (LOW/HIGH risk).
    * **Latency Monitoring:** Track response time in milliseconds.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# ============================================================
# 4. Request/Response Models
# ============================================================
class PredictionRequest(BaseModel):
    temperature: Annotated[float, Field(ge=-50, le=300, description="Equipment temperature in Celsius (°C)")]
    vibration: Annotated[float, Field(ge=0, le=100, description="Vibration level in mm/s")]
    pressure: Annotated[float, Field(ge=0, le=500, description="Pressure in PSI")]
    rpm: Annotated[float, Field(ge=0, le=50000, description="Rotational speed in RPM")]
    age_days: Annotated[int, Field(ge=0, description="Days since last maintenance")]

    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 75.5,
                "vibration": 12.0,
                "pressure": 110.0,
                "rpm": 3000,
                "age_days": 45
            }
        }

class PredictionResponse(BaseModel):
    will_fail: bool = Field(..., description="Binary indicator of failure risk")
    probability: float = Field(..., description="Calculated failure probability")
    recommendation: str = Field(..., description="Actionable maintenance advice")
    latency_ms: float = Field(..., description="API inference time in milliseconds")
    timestamp: str = Field(..., description="UTC timestamp of the prediction")
    risk_level: str = Field(..., description="Risk category (LOW/HIGH)")

# ============================================================
# 5. Endpoints
# ============================================================

@app.get("/", tags=["System Status"], summary="API Root")
def root():
    """Returns basic API information."""
    return {"message": "Predictive Maintenance API is running", "documentation": "/docs"}

@app.get("/health", tags=["System Status"], summary="Check API & Model Health")
def health_check():
    """Verify if the API is running and the model is loaded."""
    return {"status": "healthy" if model else "unhealthy", "model_loaded": model is not None}

@app.post(
    "/predict", 
    tags=["Predictions"], 
    response_model=PredictionResponse,
    summary="Get Equipment Failure Prediction",
    responses={
        503: {"description": "Model is not loaded or unavailable"},
        500: {"description": "Internal prediction error"}
    }
)
def predict(request: PredictionRequest):
    """Submits input features to the predictive maintenance model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    start_time = time.perf_counter()
    input_df = pd.DataFrame([request.model_dump()])
    
    # Prediction logic
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(input_df)[0][1])
    else:
        prob = float(model.predict(input_df)[0])
        
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return PredictionResponse(
        will_fail=prob >= 0.5,
        probability=round(prob, 4),
        recommendation="Action Required" if prob >= 0.5 else "Monitor Condition",
        latency_ms=round(latency_ms, 2),
        timestamp=datetime.utcnow().isoformat(),
        risk_level="HIGH" if prob >= 0.5 else "LOW"
    )
