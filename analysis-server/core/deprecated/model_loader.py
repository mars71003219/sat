from typing import Dict, Any, Optional
from collections import OrderedDict
import threading
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.base_model import BaseModel
from core.model_factory import model_factory
from shared.config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """
    모델 로더 - LRU 캐싱으로 메모리 효율적 관리
    팩토리 패턴을 사용하여 모델 동적 생성
    """
    
    def __init__(self, max_loaded: int = None):
        self.max_loaded = max_loaded or settings.MODEL_MAX_LOADED
        self.loaded_models: OrderedDict[str, BaseModel] = OrderedDict()
        self.lock = threading.Lock()
    
    def load_model(self, model_name: str, config: Dict[str, Any]) -> BaseModel:
        """모델 로드 (팩토리 패턴 사용)"""
        cache_key = f"{model_name}_{hash(frozenset(config.items()))}"
        
        with self.lock:
            # 캐시된 모델 확인
            if cache_key in self.loaded_models:
                logger.info(f"Model '{model_name}' found in cache")
                self.loaded_models.move_to_end(cache_key)
                return self.loaded_models[cache_key]
            
            # LRU 캐시 관리
            if len(self.loaded_models) >= self.max_loaded:
                oldest_key, oldest_model = self.loaded_models.popitem(last=False)
                logger.info(f"Evicting model from cache: {oldest_key}")
                oldest_model.unload()
            
            # 팩토리로 새 모델 생성
            logger.info(f"Loading model '{model_name}' with factory")
            model = model_factory.create(model_name, config)
            model.load()
            
            self.loaded_models[cache_key] = model
            logger.info(f"Model '{model_name}' loaded successfully (cache: {len(self.loaded_models)}/{self.max_loaded})")
            
            return model
    
    def get_model(self, model_name: str) -> Optional[BaseModel]:
        """캐시된 모델 가져오기"""
        with self.lock:
            for key, model in self.loaded_models.items():
                if key.startswith(model_name):
                    self.loaded_models.move_to_end(key)
                    return model
            return None
    
    def unload_model(self, model_name: str) -> bool:
        """특정 모델 언로드"""
        with self.lock:
            keys_to_remove = [k for k in self.loaded_models.keys() if k.startswith(model_name)]
            for key in keys_to_remove:
                model = self.loaded_models.pop(key)
                model.unload()
                logger.info(f"Model unloaded: {key}")
            return len(keys_to_remove) > 0
    
    def list_loaded_models(self) -> list:
        """로드된 모델 목록"""
        with self.lock:
            return [k.split('_')[0] for k in self.loaded_models.keys()]
    
    def get_cache_info(self) -> Dict[str, Any]:
        """캐시 정보"""
        with self.lock:
            return {
                "loaded_count": len(self.loaded_models),
                "max_loaded": self.max_loaded,
                "models": list(self.loaded_models.keys())
            }


model_loader = ModelLoader()
