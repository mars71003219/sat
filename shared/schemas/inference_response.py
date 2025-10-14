from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class InferenceResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Job status")
    message: str = Field(default="", description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_123abc",
                "status": "pending",
                "message": "Inference job submitted successfully"
            }
        }


class InferenceResult(BaseModel):
    job_id: str
    status: JobStatus
    model_name: str
    predictions: List[float] = Field(default=[], description="Predicted values")
    confidence: Optional[List[float]] = Field(default=None, description="Confidence intervals")
    metrics: Optional[Dict[str, float]] = Field(default={}, description="Performance metrics")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_123abc",
                "status": "completed",
                "model_name": "lstm_timeseries",
                "predictions": [4.8, 5.2, 5.5, 5.3, 5.7],
                "confidence": [0.95, 0.94, 0.92, 0.93, 0.91],
                "metrics": {
                    "inference_time": 0.125,
                    "mse": 0.032
                },
                "created_at": "2025-01-15T10:30:00Z",
                "completed_at": "2025-01-15T10:30:01Z"
            }
        }


class ModelInfo(BaseModel):
    name: str
    description: str
    status: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
