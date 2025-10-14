import numpy as np
from typing import Any, Dict, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from models.base_model import BaseModel
from core.model_factory import model_factory


@model_factory.register_decorator("moving_average")
class MovingAverageModel(BaseModel):
    """통계 기반 이동평균 예측 모델"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.window_size = config.get("window_size", 5)
        self.forecast_steps = config.get("forecast_steps", 5)
        self.include_trend = config.get("include_trend", True)
    
    def load(self):
        """통계 모델은 로딩 불필요"""
        self.is_loaded = True
    
    def preprocess(self, data: List[float]) -> np.ndarray:
        """데이터 전처리"""
        return np.array(data, dtype=np.float32)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """이동평균 및 트렌드 기반 예측"""
        predictions = []
        current_data = data.copy()
        
        for _ in range(self.forecast_steps):
            window = current_data[-self.window_size:] if len(current_data) >= self.window_size else current_data
            
            ma_pred = np.mean(window)
            
            if self.include_trend and len(current_data) >= 2:
                trend = (current_data[-1] - current_data[-min(len(current_data), self.window_size)]) / min(len(current_data), self.window_size)
                ma_pred += trend
            
            predictions.append(ma_pred)
            current_data = np.append(current_data, ma_pred)
        
        return np.array(predictions)
    
    def postprocess(self, predictions: np.ndarray) -> Dict[str, Any]:
        """예측 결과 후처리"""
        base_confidence = 0.90
        confidence = [base_confidence - (i * 0.03) for i in range(len(predictions))]
        
        std_dev = np.std(predictions) if len(predictions) > 1 else 0.1
        upper_bound = predictions + (1.96 * std_dev)
        lower_bound = predictions - (1.96 * std_dev)
        
        return {
            "predictions": predictions.tolist(),
            "confidence": confidence,
            "upper_bound": upper_bound.tolist(),
            "lower_bound": lower_bound.tolist(),
            "model_type": "Moving Average",
            "window_size": self.window_size,
            "forecast_steps": self.forecast_steps
        }
    
    @classmethod
    def get_description(cls) -> str:
        return "트렌드 분석을 포함한 통계 기반 이동평균 예측 모델"
    
    @classmethod
    def get_input_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "시계열 데이터 포인트"
                },
                "config": {
                    "type": "object",
                    "properties": {
                        "forecast_steps": {"type": "integer", "default": 5},
                        "window_size": {"type": "integer", "default": 5},
                        "include_trend": {"type": "boolean", "default": True}
                    }
                }
            }
        }
    
    @classmethod
    def get_output_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "predictions": {"type": "array", "items": {"type": "number"}},
                "confidence": {"type": "array", "items": {"type": "number"}},
                "upper_bound": {"type": "array", "items": {"type": "number"}},
                "lower_bound": {"type": "array", "items": {"type": "number"}}
            }
        }
