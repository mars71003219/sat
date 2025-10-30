from celery import Celery
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.config.settings import settings

celery_app = Celery(
    "operation_server",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    task_routes={
        'analysis_server.tasks.run_inference': {'queue': 'inference'},
    },
    # RabbitMQ 안정성 설정
    task_acks_late=settings.CELERY_TASK_ACKS_LATE,
    worker_prefetch_multiplier=settings.CELERY_WORKER_PREFETCH_MULTIPLIER,
    task_reject_on_worker_lost=settings.CELERY_TASK_REJECT_ON_WORKER_LOST,
    # 재시도 정책
    task_default_retry_delay=60,  # 60초 후 재시도
    task_max_retries=3,  # 최대 3번 재시도
    # 결과 만료
    result_expires=3600,  # 1시간 후 결과 삭제
)


def submit_inference_task(job_id: str, model_name: str, data: list, config: dict, metadata: dict):
    """Submit inference task to analysis server"""
    result = celery_app.send_task(
        'analysis_server.tasks.run_inference',
        args=[job_id, model_name, data, config, metadata],
        queue='inference',
        task_id=job_id
    )
    return result
