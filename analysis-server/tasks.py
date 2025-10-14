from celery import Celery, Task
import time
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.config.settings import settings
from shared.schemas import JobStatus
from core.model_loader import model_loader
from core.batch_manager import batch_manager
from utils.logger import get_logger

logger = get_logger(__name__)

celery_app = Celery(
    "analysis_server",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_track_started=settings.CELERY_TASK_TRACK_STARTED,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
)

# 모델 등록 (팩토리 패턴)
from models.timeseries import LSTMTimeSeriesModel, MovingAverageModel

logger.info("Analysis Server: Models registered via factory pattern")


def execute_inference(job_id: str, model_name: str, data: list, config: dict, metadata: dict):
    """실제 추론 실행 함수"""
    try:
        import redis
        import json
        import psycopg2
        from kafka import KafkaProducer
        
        r = redis.from_url(settings.REDIS_URL, decode_responses=True)
        r.setex(f"job:status:{job_id}", 3600, json.dumps({"status": "running"}))
        
        start_time = time.time()
        
        logger.info(f"[Worker] Job {job_id}: Loading model '{model_name}'")
        model = model_loader.load_model(model_name, config)
        
        logger.info(f"[Worker] Job {job_id}: Running inference")
        results = model.infer(data)
        
        inference_time = time.time() - start_time
        
        result_data = {
            "job_id": job_id,
            "status": "completed",
            "model_name": model_name,
            "predictions": results.get("predictions", []),
            "confidence": results.get("confidence"),
            "metrics": {
                "inference_time": inference_time,
                **results.get("metrics", {})
            },
            "metadata": {
                **metadata,
                **results,
                "processed_by": "analysis_server",
                "batch_processed": True
            },
            "created_at": metadata.get("created_at"),
            "completed_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"[Worker] Job {job_id}: Storing results")
        r.setex(f"job:result:{job_id}", 3600, json.dumps(result_data))
        r.setex(f"job:status:{job_id}", 3600, json.dumps({"status": "completed"}))
        
        # PostgreSQL 저장
        try:
            conn = psycopg2.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                database=settings.POSTGRES_DB
            )
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO inference_results 
                (job_id, model_name, status, predictions, confidence, metrics, metadata, created_at, completed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (job_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    predictions = EXCLUDED.predictions,
                    confidence = EXCLUDED.confidence,
                    metrics = EXCLUDED.metrics,
                    completed_at = EXCLUDED.completed_at
            """, (
                job_id,
                model_name,
                "completed",
                json.dumps(result_data.get("predictions", [])),
                json.dumps(result_data.get("confidence")),
                json.dumps(result_data.get("metrics", {})),
                json.dumps(result_data.get("metadata", {})),
                result_data["created_at"],
                result_data.get("completed_at")
            ))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"PostgreSQL error: {e}")
        
        # Kafka 전송
        try:
            producer = KafkaProducer(
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS.split(','),
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            producer.send(settings.KAFKA_TOPIC_INFERENCE_RESULTS, value=result_data)
            producer.flush()
            producer.close()
        except Exception as e:
            logger.error(f"Kafka error: {e}")
        
        logger.info(f"[Worker] Job {job_id}: Completed in {inference_time:.2f}s")
        
        return result_data
        
    except Exception as e:
        logger.error(f"[Worker] Job {job_id}: Error: {str(e)}")
        try:
            r.setex(f"job:status:{job_id}", 3600, json.dumps({"status": "failed", "error": str(e)}))
        except:
            pass
        raise


class InferenceTask(Task):
    """Celery 태스크 클래스"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed: {exc}")
        try:
            import redis
            r = redis.from_url(settings.REDIS_URL, decode_responses=True)
            r.setex(f"job:status:{task_id}", 3600, f'{{"status": "failed", "error": "{str(exc)}"}}')
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")
    
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Task {task_id} completed successfully")


@celery_app.task(bind=True, base=InferenceTask, name='analysis_server.tasks.run_inference')
def run_inference(self, job_id: str, model_name: str, data: list, config: dict, metadata: dict):
    """
    Celery 태스크 - 배치 큐에 추가
    일정 크기 또는 시간이 되면 배치 처리
    """
    logger.info(f"[Task] Received job {job_id} for model '{model_name}'")
    
    # 배치 큐에 추가
    batch_manager.add_to_batch(
        model_name=model_name,
        job_id=job_id,
        data=data,
        config=config,
        metadata=metadata,
        batch_callback=execute_inference
    )
    
    return {"job_id": job_id, "status": "queued_for_batch"}
