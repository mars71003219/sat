import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from shared.config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class PostgresClient:
    """PostgreSQL client for persistent storage"""
    
    def __init__(self):
        self.conn_params = {
            "host": settings.POSTGRES_HOST,
            "port": settings.POSTGRES_PORT,
            "user": settings.POSTGRES_USER,
            "password": settings.POSTGRES_PASSWORD,
            "database": settings.POSTGRES_DB
        }
    
    @contextmanager
    def get_connection(self):
        """Get database connection context manager"""
        conn = None
        try:
            conn = psycopg2.connect(**self.conn_params)
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def init_tables(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inference_results (
                    id SERIAL PRIMARY KEY,
                    job_id VARCHAR(255) UNIQUE NOT NULL,
                    model_name VARCHAR(255) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    predictions JSONB,
                    confidence JSONB,
                    metrics JSONB,
                    metadata JSONB,
                    error_message TEXT,
                    created_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    INDEX idx_job_id (job_id),
                    INDEX idx_model_name (model_name),
                    INDEX idx_created_at (created_at)
                )
            """)
            
            logger.info("Database tables initialized")
    
    def save_result(self, result: Dict[str, Any]):
        """Save inference result"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO inference_results 
                (job_id, model_name, status, predictions, confidence, metrics, metadata, error_message, created_at, completed_at)
                VALUES (%(job_id)s, %(model_name)s, %(status)s, %(predictions)s, %(confidence)s, %(metrics)s, %(metadata)s, %(error_message)s, %(created_at)s, %(completed_at)s)
                ON CONFLICT (job_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    predictions = EXCLUDED.predictions,
                    confidence = EXCLUDED.confidence,
                    metrics = EXCLUDED.metrics,
                    error_message = EXCLUDED.error_message,
                    completed_at = EXCLUDED.completed_at
            """, {
                "job_id": result["job_id"],
                "model_name": result["model_name"],
                "status": result["status"],
                "predictions": psycopg2.extras.Json(result.get("predictions", [])),
                "confidence": psycopg2.extras.Json(result.get("confidence")),
                "metrics": psycopg2.extras.Json(result.get("metrics", {})),
                "metadata": psycopg2.extras.Json(result.get("metadata", {})),
                "error_message": result.get("error_message"),
                "created_at": result["created_at"],
                "completed_at": result.get("completed_at")
            })
            
            logger.debug(f"Saved result for job {result['job_id']}")
    
    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get inference result by job ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                "SELECT * FROM inference_results WHERE job_id = %s",
                (job_id,)
            )
            result = cursor.fetchone()
            return dict(result) if result else None
    
    def get_results_history(self, limit: int = 100, offset: int = 0, model_name: str = None) -> List[Dict[str, Any]]:
        """Get inference results history"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            query = "SELECT * FROM inference_results"
            params = []

            if model_name:
                query += " WHERE model_name = %s"
                params.append(model_name)

            query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])

            cursor.execute(query, params)
            results = cursor.fetchall()
            return [dict(row) for row in results]

    def get_recent_results(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent inference results"""
        return self.get_results_history(limit=limit)

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT
                    COUNT(*) as total_jobs,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed,
                    ROUND(
                        100.0 * COUNT(CASE WHEN status = 'completed' THEN 1 END) / NULLIF(COUNT(*), 0),
                        2
                    ) as success_rate
                FROM inference_results
            """)

            stats = dict(cursor.fetchone())

            # Get models usage
            cursor.execute("""
                SELECT model_name, COUNT(*) as count
                FROM inference_results
                GROUP BY model_name
                ORDER BY count DESC
            """)

            models = cursor.fetchall()
            stats['models_used'] = {row['model_name']: row['count'] for row in models}

            return stats

    def update_job_status(self, job_id: str, status: str):
        """Update job status"""
        from datetime import datetime
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO inference_results (job_id, model_name, status, created_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (job_id) DO UPDATE SET status = EXCLUDED.status
            """, (job_id, 'unknown', status, datetime.utcnow()))
            logger.debug(f"Updated job {job_id} status to {status}")
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT job_id, model_name, status, created_at, completed_at, error_message
                FROM inference_results WHERE job_id = %s
            """, (job_id,))
            result = cursor.fetchone()
            if result:
                return dict(result)
            return {"status": "not_found"}
    
    def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job result"""
        return self.get_result(job_id)


    def query(self, sql: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute a query and return results"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(sql, params or ())
            results = cursor.fetchall()
            return [dict(row) for row in results]

    @staticmethod
    def _get_timestamp():
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now()

    @staticmethod
    def _parse_timestamp(timestamp_str: str):
        """Parse timestamp string"""
        from datetime import datetime
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            return datetime.now()


postgres_client = PostgresClient()


def save_inference_result(result: Dict[str, Any]):
    """Helper function to save inference result"""
    try:
        postgres_client.save_result(result)
    except Exception as e:
        logger.error(f"Failed to save result to PostgreSQL: {e}")


def get_inference_result(job_id: str) -> Optional[Dict[str, Any]]:
    """Helper function to get inference result"""
    try:
        return postgres_client.get_result(job_id)
    except Exception as e:
        logger.error(f"Failed to get result from PostgreSQL: {e}")
        return None


# ============================================================
# 새로운 스키마를 위한 함수들
# ============================================================

def save_inference_job(job_id: str, satellite_id: str, source: str, 
                      trigger_reason: str, total_subsystems: int, metadata: dict = None):
    """추론 작업 생성"""
    client = PostgresClient()
    with client.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO inference_jobs 
            (job_id, satellite_id, source, trigger_reason, total_subsystems, metadata, status)
            VALUES (%s, %s, %s, %s, %s, %s, 'processing')
            ON CONFLICT (job_id) DO NOTHING
        """, (job_id, satellite_id, source, trigger_reason, total_subsystems, 
              psycopg2.extras.Json(metadata) if metadata else None))


def save_subsystem_inference_start(job_id: str, subsystem: str, model_name: str,
                                   input_data: list, input_features: list, metadata: dict):
    """서브시스템 추론 시작 기록"""
    client = PostgresClient()
    
    # Job이 없으면 생성
    satellite_id = metadata.get('satellite_id', 'UNKNOWN')
    source = metadata.get('source', 'unknown')
    trigger_reason = metadata.get('trigger_reason', 'unknown')
    
    with client.get_connection() as conn:
        cursor = conn.cursor()
        
        # Job 생성 (이미 있으면 무시)
        cursor.execute("""
            INSERT INTO inference_jobs 
            (job_id, satellite_id, source, trigger_reason, status, total_subsystems, metadata)
            VALUES (%s, %s, %s, %s, 'processing', 4, %s)
            ON CONFLICT (job_id) DO UPDATE 
            SET total_subsystems = GREATEST(inference_jobs.total_subsystems, 4)
        """, (job_id, satellite_id, source, trigger_reason, 
              psycopg2.extras.Json(metadata)))
        
        # 서브시스템 추론 레코드 생성
        cursor.execute("""
            INSERT INTO subsystem_inferences
            (job_id, subsystem, model_name, status, input_data, input_features)
            VALUES (%s, %s, %s, 'processing', %s, %s)
        """, (job_id, subsystem, model_name, 
              psycopg2.extras.Json(input_data),
              input_features))


def save_subsystem_inference_result(job_id: str, subsystem: str, model_name: str,
                                    status: str, input_data: list = None, 
                                    input_features: list = None,
                                    predictions: list = None, confidence: list = None,
                                    anomaly_score: float = None, anomaly_detected: bool = False,
                                    metrics: dict = None, error_message: str = None):
    """서브시스템 추론 결과 저장"""
    import psycopg2.extras
    from datetime import datetime
    
    client = PostgresClient()
    with client.get_connection() as conn:
        cursor = conn.cursor()
        
        # 통계 계산
        import numpy as np
        input_mean = float(np.mean(input_data)) if input_data else None
        input_std = float(np.std(input_data)) if input_data else None
        pred_mean = float(np.mean(predictions)) if predictions else None
        pred_std = float(np.std(predictions)) if predictions else None

        # Convert numpy types to Python native types
        if isinstance(anomaly_detected, np.bool_):
            anomaly_detected = bool(anomaly_detected)

        cursor.execute("""
            UPDATE subsystem_inferences
            SET status = %s,
                predictions = %s,
                confidence = %s,
                anomaly_score = %s,
                anomaly_detected = %s,
                inference_time_ms = %s,
                model_type = %s,
                sequence_length = %s,
                forecast_horizon = %s,
                input_mean = %s,
                input_std = %s,
                prediction_mean = %s,
                prediction_std = %s,
                processed_by = %s,
                completed_at = %s,
                error_message = %s
            WHERE job_id = %s AND subsystem = %s
        """, (
            status,
            psycopg2.extras.Json(predictions) if predictions else None,
            psycopg2.extras.Json(confidence) if confidence else None,
            anomaly_score,
            anomaly_detected,
            metrics.get('inference_time') * 1000 if metrics and 'inference_time' in metrics else None,
            metrics.get('model_type') if metrics else None,
            metrics.get('sequence_length') if metrics else None,
            metrics.get('forecast_steps') if metrics else None,
            input_mean,
            input_std,
            pred_mean,
            pred_std,
            'triton_server',
            datetime.now() if status == 'completed' else None,
            error_message,
            job_id,
            subsystem
        ))
        
        # Job 완료 상태 업데이트
        if status == 'completed':
            cursor.execute("""
                UPDATE inference_jobs
                SET completed_subsystems = completed_subsystems + 1,
                    status = CASE 
                        WHEN completed_subsystems + 1 >= total_subsystems THEN 'completed'
                        ELSE 'processing'
                    END,
                    completed_at = CASE
                        WHEN completed_subsystems + 1 >= total_subsystems THEN NOW()
                        ELSE completed_at
                    END
                WHERE job_id = %s
            """, (job_id,))


def get_recent_anomalies(limit: int = 10):
    """최근 이상 감지 결과 조회"""
    client = PostgresClient()
    with client.get_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT * FROM v_anomaly_alerts
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
        return cursor.fetchall()


def get_job_summary(job_id: str):
    """작업 요약 조회"""
    client = PostgresClient()
    with client.get_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT * FROM v_inference_job_summary
            WHERE job_id = %s
        """, (job_id,))
        return cursor.fetchone()
