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
    }
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
