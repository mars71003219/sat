"""
Analysis Server - Celery Tasks (Triton Client)
Triton Inference Server를 사용한 추론 작업
"""
from celery import Celery, Task
import time
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.config.settings import settings
from shared.schemas import JobStatus
from core.triton_client import triton_client
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

logger.info("Analysis Server: Using Triton Inference Server")


def execute_inference(job_id: str, model_name: str, data: list, config: dict, metadata: dict):
    """
    Triton을 사용한 추론 실행

    Args:
        job_id: 작업 ID
        model_name: 모델 이름 ("lstm_timeseries" 또는 "moving_average")
        data: 입력 데이터
        config: 모델 설정
        metadata: 메타데이터

    Returns:
        결과 딕셔너리
    """
    try:
        import redis
        import json
        import psycopg2
        from kafka import KafkaProducer

        r = redis.from_url(settings.REDIS_URL, decode_responses=True)
        r.setex(f"job:status:{job_id}", 3600, json.dumps({"status": "running"}))

        start_time = time.time()

        # ========================================
        # Triton 추론 실행 (전처리 + 추론 + 후처리 모두 포함)
        # ========================================
        logger.info(f"[Worker] Job {job_id}: Calling Triton for model '{model_name}'")

        result = triton_client.infer(
            model_name=model_name,
            data=data,
            config=config
        )

        inference_time = time.time() - start_time

        logger.info(f"[Worker] Job {job_id}: Triton inference completed in {inference_time:.3f}s")

        # ========================================
        # 결과 데이터 구성
        # ========================================
        result_data = {
            "job_id": job_id,
            "status": "completed",
            "model_name": model_name,
            "predictions": result.get("predictions", []),
            "confidence": result.get("confidence"),
            "metrics": {
                "inference_time": inference_time,
                **result.get("metrics", {})
            },
            "metadata": {
                **metadata,
                "processed_by": "triton_server",
                "triton_client": "grpc"
            },
            "created_at": metadata.get("created_at"),
            "completed_at": datetime.utcnow().isoformat()
        }

        # upper_bound, lower_bound 추가 (Moving Average의 경우)
        if "upper_bound" in result:
            result_data["upper_bound"] = result["upper_bound"]
        if "lower_bound" in result:
            result_data["lower_bound"] = result["lower_bound"]

        # ========================================
        # 결과 저장 - Redis
        # ========================================
        logger.info(f"[Worker] Job {job_id}: Storing results to Redis")
        r.setex(f"job:result:{job_id}", 3600, json.dumps(result_data))
        r.setex(f"job:status:{job_id}", 3600, json.dumps({"status": "completed"}))

        # ========================================
        # 결과 저장 - PostgreSQL
        # ========================================
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
            logger.info(f"[Worker] Job {job_id}: Saved to PostgreSQL")
        except Exception as e:
            logger.error(f"[Worker] Job {job_id}: PostgreSQL error: {e}")

        # ========================================
        # 결과 전송 - Kafka
        # ========================================
        try:
            producer = KafkaProducer(
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS.split(','),
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            producer.send(settings.KAFKA_TOPIC_INFERENCE_RESULTS, value=result_data)
            producer.flush()
            producer.close()
            logger.info(f"[Worker] Job {job_id}: Sent to Kafka")
        except Exception as e:
            logger.error(f"[Worker] Job {job_id}: Kafka error: {e}")

        logger.info(f"[Worker] Job {job_id}: All tasks completed")

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
    Celery 태스크 - Triton을 통한 추론 실행

    배치 처리는 Triton Server의 Dynamic Batching이 자동으로 수행합니다.
    따라서 기존 BatchManager는 불필요합니다.

    Args:
        job_id: 작업 ID
        model_name: 모델 이름
        data: 입력 데이터
        config: 모델 설정
        metadata: 메타데이터

    Returns:
        결과 딕셔너리
    """
    logger.info(f"[Task] Received job {job_id} for model '{model_name}'")

    # Triton이 자동으로 배치 처리하므로, 직접 추론 실행
    return execute_inference(job_id, model_name, data, config, metadata)
