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
from core.triton_client import get_triton_client
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
    # RabbitMQ 안정성 설정
    task_acks_late=settings.CELERY_TASK_ACKS_LATE,
    worker_prefetch_multiplier=settings.CELERY_WORKER_PREFETCH_MULTIPLIER,
    task_reject_on_worker_lost=settings.CELERY_TASK_REJECT_ON_WORKER_LOST,
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
        import json
        import psycopg2
        from kafka import KafkaProducer

        start_time = time.time()

        # PostgreSQL에 작업 상태 업데이트 (running)
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
                INSERT INTO inference_results (job_id, status, created_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (job_id) DO UPDATE SET status = EXCLUDED.status
            """, (job_id, "running", datetime.utcnow().isoformat()))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.warning(f"[Worker] Job {job_id}: Failed to update status to running: {e}")

        # ========================================
        # Triton 추론 실행 (전처리 + 추론 + 후처리 모두 포함)
        # ========================================
        logger.info(f"[Worker] Job {job_id}: Calling Triton for model '{model_name}'")

        client = get_triton_client()
        result = client.infer(
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
        # 결과 저장 - PostgreSQL
        # ========================================
        logger.info(f"[Worker] Job {job_id}: Storing results to PostgreSQL")
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
        # PostgreSQL에 실패 상태 업데이트
        try:
            import psycopg2
            import json
            conn = psycopg2.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                database=settings.POSTGRES_DB
            )
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE inference_results
                SET status = %s, metadata = %s
                WHERE job_id = %s
            """, ("failed", json.dumps({"error": str(e)}), job_id))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as db_err:
            logger.error(f"[Worker] Job {job_id}: Failed to update error status: {db_err}")
        raise


class InferenceTask(Task):
    """Celery 태스크 클래스 - RabbitMQ에서 안정적으로 동작"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed: {exc}")
        # 실패 정보는 이미 execute_inference에서 PostgreSQL에 저장됨
        logger.info(f"Task {task_id} failure logged to PostgreSQL")

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


@celery_app.task(bind=True, base=InferenceTask, name='analysis_server.tasks.run_subsystem_inference')
def run_subsystem_inference(self, job_id: str, subsystem: str, model_name: str, 
                            input_data: list, input_features: list, 
                            config: dict, metadata: dict):
    """
    서브시스템별 Celery 태스크 - Triton을 통한 추론 실행
    
    Args:
        job_id: 전체 작업 ID (여러 서브시스템을 묶는 단위)
        subsystem: 서브시스템 이름 ('eps', 'thermal', 'aocs', 'comm')
        model_name: 모델 이름 ('lstm_eps', 'lstm_thermal', etc.)
        input_data: 입력 데이터 리스트
        input_features: 특징 이름 리스트
        config: 모델 설정
        metadata: 메타데이터 (satellite_id, source, trigger_reason 등)
    
    Returns:
        결과 딕셔너리
    """
    task_id = self.request.id
    logger.info(f"[Subsystem Task] {subsystem} inference started for job {job_id}")
    
    # PostgreSQL에 작업 시작 기록 (새 스키마 사용)
    from shared.database.postgres_client import save_subsystem_inference_start, save_subsystem_inference_result
    
    try:
        # 1. 작업 시작 기록
        save_subsystem_inference_start(job_id, subsystem, model_name, input_data, 
                                      input_features, metadata)
        
        # 2. Triton 추론 실행 (기존 execute_inference 재사용)
        result = execute_inference(
            job_id=f"{job_id}_{subsystem}",
            model_name=model_name,
            data=input_data,
            config=config,
            metadata={**metadata, 'subsystem': subsystem, 'features': input_features}
        )
        
        # 3. 이상 감지 점수 계산 (간단한 예시)
        anomaly_score = calculate_anomaly_score(input_data, result.get('predictions', []))
        anomaly_detected = anomaly_score > 0.7  # 임계값
        
        # 4. 결과 저장
        save_subsystem_inference_result(
            job_id=job_id,
            subsystem=subsystem,
            model_name=model_name,
            status='completed',
            predictions=result.get('predictions'),
            confidence=result.get('confidence'),
            anomaly_score=anomaly_score,
            anomaly_detected=anomaly_detected,
            metrics=result.get('metrics'),
            input_data=input_data,
            input_features=input_features
        )
        
        logger.info(f"[Subsystem Task] {subsystem} inference completed for job {job_id}")
        
        # Convert numpy types for JSON serialization
        import numpy as np
        if isinstance(anomaly_detected, np.bool_):
            anomaly_detected = bool(anomaly_detected)
        if isinstance(anomaly_score, (np.integer, np.floating)):
            anomaly_score = float(anomaly_score)
        
        return {
            'job_id': job_id,
            'subsystem': subsystem,
            'model_name': model_name,
            'status': 'completed',
            'anomaly_detected': anomaly_detected,
            'anomaly_score': anomaly_score,
            **result
        }
        
    except Exception as e:
        logger.error(f"[Subsystem Task] {subsystem} inference failed: {e}", exc_info=True)
        
        # 실패 기록
        try:
            save_subsystem_inference_result(
                job_id=job_id,
                subsystem=subsystem,
                model_name=model_name,
                status='failed',
                error_message=str(e),
                input_data=input_data,
                input_features=input_features
            )
        except:
            pass
        
        raise


def calculate_anomaly_score(input_data: list, predictions: list) -> float:
    """
    간단한 이상 점수 계산
    실제로는 더 복잡한 통계적 방법이나 ML 모델 사용
    """
    try:
        if not input_data or not predictions:
            return 0.0
        
        import numpy as np
        
        # 예측 오차 기반 이상 점수
        input_mean = np.mean(input_data[-len(predictions):]) if len(input_data) >= len(predictions) else np.mean(input_data)
        pred_mean = np.mean(predictions)
        
        # 정규화된 차이
        if input_mean != 0:
            diff_ratio = abs(pred_mean - input_mean) / abs(input_mean)
            score = min(diff_ratio, 1.0)
        else:
            score = 0.0
        
        return round(score, 4)
        
    except Exception as e:
        logger.error(f"Error calculating anomaly score: {e}")
        return 0.0
