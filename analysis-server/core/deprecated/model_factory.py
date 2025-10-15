from typing import Dict, Type, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.base_model import BaseModel
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """
    팩토리 패턴 - 모델 동적 생성 및 관리
    새로운 모델 추가 시 register 메서드만 호출하면 됨
    """
    
    _models: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, model_name: str, model_class: Type[BaseModel]):
        """모델 클래스 등록"""
        cls._models[model_name] = model_class
        logger.info(f"Model registered: {model_name} -> {model_class.__name__}")
    
    @classmethod
    def register_decorator(cls, model_name: str):
        """데코레이터 방식 모델 등록"""
        def decorator(model_class: Type[BaseModel]):
            cls.register(model_name, model_class)
            return model_class
        return decorator
    
    @classmethod
    def create(cls, model_name: str, config: Dict[str, Any]) -> BaseModel:
        """모델 인스턴스 생성"""
        if model_name not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(f"Model '{model_name}' not found. Available: {available}")
        
        model_class = cls._models[model_name]
        model = model_class(config)
        logger.info(f"Model instance created: {model_name}")
        return model
    
    @classmethod
    def get_registered_models(cls) -> list:
        """등록된 모델 목록 반환"""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """모델 정보 반환"""
        if model_name not in cls._models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model_class = cls._models[model_name]
        return {
            "name": model_name,
            "class": model_class.__name__,
            "description": model_class.get_description(),
            "input_schema": model_class.get_input_schema(),
            "output_schema": model_class.get_output_schema()
        }


model_factory = ModelFactory()
