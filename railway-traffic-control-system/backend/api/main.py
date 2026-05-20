import os
import sys
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, Header, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Add backend dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas.models import (
    ConflictRequest, ConflictResponse, DelayRequest, DelayResponse,
    CombinedResponse, SimulationRequest, OptimizationRequest, OptimizationResponse,
    KPIResponse
)
from api.state import state_manager
from conflict_detector_onnx import ConflictDetector
from delay_predictor_onnx import DelayPredictor
from throughput_optimizer_ortools import ThroughputOptimizer

load_dotenv()

app = FastAPI(
    title="Railway Traffic Control System API",
    version="2.0.0",
    description="Next-gen Railway Traffic Control API using FastAPI, ONNX, and OR-Tools"
)

# CORS
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv('ALLOWED_ORIGINS', 'http://localhost:8501,http://127.0.0.1:8501,http://localhost:8000').split(',')]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Security
API_KEY = os.getenv('API_KEY', 'test-api-key')

async def verify_api_key(x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return x_api_key

# Models
try:
    conflict_model_path = os.getenv('CONFLICT_MODEL_PATH', 'backend/models/conflict_detector.onnx')
    delay_model_path = os.getenv('DELAY_MODEL_PATH', 'backend/models/delay_predictor.onnx')

    conflict_detector = ConflictDetector(conflict_model_path)
    delay_predictor = DelayPredictor(delay_model_path)
    throughput_optimizer = ThroughputOptimizer()
except Exception as e:
    print(f"Error loading models: {e}")
    conflict_detector = None
    delay_predictor = None
    throughput_optimizer = None

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "engines": {
            "conflict_detector": conflict_detector is not None,
            "delay_predictor": delay_predictor is not None,
            "optimizer": throughput_optimizer is not None
        }
    }

@app.post("/api/predict/conflict", response_model=ConflictResponse)
async def predict_conflict(request: ConflictRequest, api_key: str = Depends(verify_api_key)):
    result = conflict_detector.predict(request.model_dump())
    await state_manager.record_prediction(result['conflict_probability'], 0.0, result['risk_level'])
    return result

@app.post("/api/predict/delay", response_model=DelayResponse)
async def predict_delay(request: DelayRequest, api_key: str = Depends(verify_api_key)):
    result = delay_predictor.predict(request.model_dump())
    await state_manager.record_prediction(0.0, result['predicted_delay_minutes'], result['severity'])
    return result

@app.post("/api/predict/combined", response_model=CombinedResponse)
async def predict_combined(request: ConflictRequest, api_key: str = Depends(verify_api_key)):
    data = request.model_dump()
    conflict_res = conflict_detector.predict(data)
    delay_res = delay_predictor.predict(data)

    await state_manager.record_prediction(
        conflict_res['conflict_probability'],
        delay_res['predicted_delay_minutes'],
        conflict_res['risk_level']
    )

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "conflict_analysis": conflict_res,
        "delay_analysis": delay_res,
        "overall_risk_assessment": {
            "conflict_risk": conflict_res['risk_level'],
            "delay_severity": delay_res['severity']
        }
    }

@app.post("/api/simulate")
async def simulate_scenario(request: SimulationRequest, api_key: str = Depends(verify_api_key)):
    return delay_predictor.simulate_scenario(request.baseline.model_dump(), request.modifications)

@app.post("/api/optimize/schedule", response_model=OptimizationResponse)
async def optimize_schedule(request: OptimizationRequest, api_key: str = Depends(verify_api_key)):
    return throughput_optimizer.optimize_train_schedule(
        [t.model_dump() for t in request.trains],
        request.platforms,
        request.section_capacity,
        request.time_horizon
    )

@app.get("/api/metrics/kpi", response_model=KPIResponse)
async def get_kpis():
    return await state_manager.get_kpis()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
