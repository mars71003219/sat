import redis
import json
from typing import Any, Dict, Optional
from shared.config.settings import settings
from shared.schemas import JobStatus
from utils.logger import get_logger

logger = get_logger(__name__)


class RedisClient:
    """Redis client for caching and job status management"""
    
    def __init__(self):
        self.client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.job_status_prefix = "job:status:"
        self.job_result_prefix = "job:result:"
        self.job_ttl = 3600  # 1 hour
    
    def set_job_status(self, job_id: str, status: JobStatus, error: str = None):
        """Set job status in Redis"""
        key = f"{self.job_status_prefix}{job_id}"
        data = {"status": status.value}
        if error:
            data["error"] = error
        self.client.setex(key, self.job_ttl, json.dumps(data))
        logger.debug(f"Set job {job_id} status to {status}")
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status from Redis"""
        key = f"{self.job_status_prefix}{job_id}"
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return {"status": "not_found"}
    
    def set_job_result(self, job_id: str, result: Dict[str, Any]):
        """Store job result in Redis"""
        key = f"{self.job_result_prefix}{job_id}"
        self.client.setex(key, self.job_ttl, json.dumps(result))
        logger.debug(f"Stored result for job {job_id}")
    
    def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job result from Redis"""
        key = f"{self.job_result_prefix}{job_id}"
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None
    
    def cache_model_state(self, model_name: str, state: Dict[str, Any], ttl: int = 3600):
        """Cache model state"""
        key = f"model:state:{model_name}"
        self.client.setex(key, ttl, json.dumps(state))
    
    def get_model_state(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get cached model state"""
        key = f"model:state:{model_name}"
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None
    
    def ping(self) -> bool:
        """Check Redis connection"""
        try:
            return self.client.ping()
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False


redis_client = RedisClient()
