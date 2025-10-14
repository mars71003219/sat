from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime


class InferenceRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to use")
    data: List[float] = Field(..., description="Time series data points")
    config: Optional[Dict[str, Any]] = Field(default={}, description="Model-specific configuration")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "lstm_timeseries",
                "data": [1.2, 2.3, 3.1, 2.8, 3.5, 4.2, 3.9, 4.5],
                "config": {
                    "forecast_steps": 5,
                    "confidence_level": 0.95
                },
                "metadata": {
                    "source": "sensor_01",
                    "timestamp": "2025-01-15T10:30:00Z"
                }
            }
        }


class ModelLoadRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to load")
    config: Optional[Dict[str, Any]] = Field(default={}, description="Model configuration")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "lstm_timeseries",
                "config": {
                    "sequence_length": 10,
                    "hidden_size": 64
                }
            }
        }
