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
