from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all AI models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.is_loaded = False
    
    @abstractmethod
    def load(self):
        """Load model weights and initialize"""
        pass
    
    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """Preprocess input data"""
        pass
    
    @abstractmethod
    def predict(self, data: Any) -> Any:
        """Run inference"""
        pass
    
    @abstractmethod
    def postprocess(self, predictions: Any) -> Dict[str, Any]:
        """Postprocess predictions"""
        pass
    
    def infer(self, data: Any) -> Dict[str, Any]:
        """Complete inference pipeline"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        preprocessed = self.preprocess(data)
        predictions = self.predict(preprocessed)
        results = self.postprocess(predictions)
        return results
    
    def unload(self):
        """Unload model from memory"""
        self.model = None
        self.is_loaded = False
    
    @classmethod
    @abstractmethod
    def get_description(cls) -> str:
        """Return model description"""
        pass
    
    @classmethod
    @abstractmethod
    def get_input_schema(cls) -> Dict[str, Any]:
        """Return input data schema"""
        pass
    
    @classmethod
    @abstractmethod
    def get_output_schema(cls) -> Dict[str, Any]:
        """Return output data schema"""
        pass
