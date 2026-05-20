from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class ConflictRequest(BaseModel):
    trains_in_section: int = Field(..., ge=0, le=200)
    available_platforms: int = Field(..., ge=0, le=50)
    platform_utilization: float = Field(..., ge=0.0, le=100.0)
    weather_severity: float = Field(..., ge=0.0, le=1.0)
    rainfall_mm: float = Field(..., ge=0.0, le=500.0)
    fog_intensity: float = Field(..., ge=0.0, le=1.0)
    temperature_c: float = Field(..., ge=-50.0, le=60.0)
    is_peak_hour: int = Field(..., ge=0, le=1)

class Recommendation(BaseModel):
    priority: str
    action: str
    details: str

class ConflictResponse(BaseModel):
    conflict_probability: float
    risk_level: str
    recommendations: List[Recommendation]
    timestamp: str

class DelayRequest(ConflictRequest):
    pass

class MitigationStrategy(BaseModel):
    strategy: str
    implementation: str
    expected_reduction: str

class DelayResponse(BaseModel):
    predicted_delay_minutes: float
    severity: str
    impact_description: str
    mitigation_strategies: List[MitigationStrategy]
    timestamp: str

class CombinedResponse(BaseModel):
    timestamp: str
    conflict_analysis: ConflictResponse
    delay_analysis: DelayResponse
    overall_risk_assessment: Dict

class SimulationRequest(BaseModel):
    baseline: ConflictRequest
    modifications: Dict

class TrainScheduleItem(BaseModel):
    id: str
    priority: int = Field(..., ge=1, le=5)
    arrival_time: int = Field(..., ge=0)
    duration: int = Field(..., ge=1)

class OptimizationRequest(BaseModel):
    trains: List[TrainScheduleItem]
    platforms: int = 3
    section_capacity: int = 25
    time_horizon: int = 60

class OptimizationResponse(BaseModel):
    status: str
    total_trains_scheduled: int
    scheduled_trains: List[Dict]
    unresolved_conflicts: int
    computation_time_ms: float

class KPIResponse(BaseModel):
    timestamp: str
    data_source: str
    sample_size: Dict
    throughput: Dict
    average_delay: Dict
    punctuality: Dict
    conflicts: Dict
