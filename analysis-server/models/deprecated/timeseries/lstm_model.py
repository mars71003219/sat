import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from models.base_model import BaseModel
from core.model_factory import model_factory


class LSTMNetwork(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


@model_factory.register_decorator("lstm_timeseries")
class LSTMTimeSeriesModel(BaseModel):
    """LSTM 기반 시계열 예측 모델"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sequence_length = config.get("sequence_length", 10)
        self.forecast_steps = config.get("forecast_steps", 5)
        self.hidden_size = config.get("hidden_size", 64)
        self.num_layers = config.get("num_layers", 2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load(self):
        """LSTM 모델 로드"""
        self.model = LSTMNetwork(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1
        ).to(self.device)
        
        self.model.eval()
        self.is_loaded = True
    
    def preprocess(self, data: List[float]) -> torch.Tensor:
        """시계열 데이터 전처리"""
        data_array = np.array(data, dtype=np.float32)
        
        self.mean = np.mean(data_array)
        self.std = np.std(data_array) + 1e-8
        normalized = (data_array - self.mean) / self.std
        
        if len(normalized) < self.sequence_length:
            padded = np.pad(normalized, (self.sequence_length - len(normalized), 0), mode='edge')
            sequences = padded.reshape(1, self.sequence_length, 1)
        else:
            sequences = normalized[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        return torch.FloatTensor(sequences).to(self.device)
    
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """LSTM 추론"""
        with torch.no_grad():
            predictions = []
            current_seq = data.clone()
            
            for _ in range(self.forecast_steps):
                pred = self.model(current_seq)
                predictions.append(pred.item())
                current_seq = torch.cat([current_seq[:, 1:, :], pred.view(1, 1, 1)], dim=1)
            
            return torch.FloatTensor(predictions)
    
    def postprocess(self, predictions: torch.Tensor) -> Dict[str, Any]:
        """예측 결과 후처리"""
        denormalized = predictions.cpu().numpy() * self.std + self.mean
        confidence = [0.95 - (i * 0.02) for i in range(len(denormalized))]
        
        return {
            "predictions": denormalized.tolist(),
            "confidence": confidence,
            "model_type": "LSTM",
            "forecast_steps": self.forecast_steps
        }
    
    @classmethod
    def get_description(cls) -> str:
        return "LSTM 기반 딥러닝 시계열 예측 모델"
    
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
                        "sequence_length": {"type": "integer", "default": 10},
                        "hidden_size": {"type": "integer", "default": 64}
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
                "confidence": {"type": "array", "items": {"type": "number"}}
            }
        }
