import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "AI Inference Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # PostgreSQL
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "admin")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "admin123")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "orders_db")

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Elasticsearch
    ELASTICSEARCH_URL: str = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_TOPIC_INFERENCE_RESULTS: str = "inference.results"
    KAFKA_TOPIC_INFERENCE_STATUS: str = "inference.status"

    # Celery (RabbitMQ로 변경 - 더 안정적인 메시지 브로커)
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "rpc://")
    CELERY_TASK_TRACK_STARTED: bool = True
    CELERY_TASK_TIME_LIMIT: int = 1800
    CELERY_TASK_SOFT_TIME_LIMIT: int = 1700

    # RabbitMQ 추가 설정 (안정성 향상)
    CELERY_TASK_ACKS_LATE: bool = True  # 작업 완료 후 ACK
    CELERY_WORKER_PREFETCH_MULTIPLIER: int = 1  # 한 번에 하나씩 처리
    CELERY_TASK_REJECT_ON_WORKER_LOST: bool = True  # Worker 손실 시 재시도

    # Model Settings
    MODEL_CACHE_DIR: str = "/app/model_cache"
    MODEL_MAX_LOADED: int = 3
    MODEL_WARMUP_ENABLED: bool = True

    # GPU Settings
    GPU_MEMORY_FRACTION: float = 0.8
    CUDA_VISIBLE_DEVICES: Optional[str] = os.getenv("CUDA_VISIBLE_DEVICES")

    # Inference Settings
    MAX_BATCH_SIZE: int = 8
    INFERENCE_TIMEOUT: int = 300

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
