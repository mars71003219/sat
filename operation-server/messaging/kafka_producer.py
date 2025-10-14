import json
from kafka import KafkaProducer
from typing import Dict, Any
from shared.config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class KafkaClient:
    """Kafka producer for async messaging"""
    
    def __init__(self):
        self.producer = None
        self._initialize_producer()
    
    def _initialize_producer(self):
        """Initialize Kafka producer"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS.split(','),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            logger.info("Kafka producer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            self.producer = None
    
    def send_message(self, topic: str, message: Dict[str, Any], key: str = None):
        """Send message to Kafka topic"""
        if not self.producer:
            logger.warning("Kafka producer not initialized, skipping message")
            return
        
        try:
            future = self.producer.send(topic, value=message, key=key)
            future.get(timeout=10)
            logger.debug(f"Sent message to topic '{topic}'")
        except Exception as e:
            logger.error(f"Failed to send message to Kafka: {e}")
    
    def close(self):
        """Close Kafka producer"""
        if self.producer:
            self.producer.close()


kafka_client = KafkaClient()


def send_inference_result(result: Dict[str, Any]):
    """Send inference result to Kafka"""
    try:
        kafka_client.send_message(
            topic=settings.KAFKA_TOPIC_INFERENCE_RESULTS,
            message=result,
            key=result.get("job_id")
        )
    except Exception as e:
        logger.error(f"Failed to send inference result to Kafka: {e}")


def send_status_update(job_id: str, status: str, metadata: Dict[str, Any] = None):
    """Send status update to Kafka"""
    try:
        message = {
            "job_id": job_id,
            "status": status,
            "metadata": metadata or {}
        }
        kafka_client.send_message(
            topic=settings.KAFKA_TOPIC_INFERENCE_STATUS,
            message=message,
            key=job_id
        )
    except Exception as e:
        logger.error(f"Failed to send status update to Kafka: {e}")
