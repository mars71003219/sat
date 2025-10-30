#!/usr/bin/env python3
"""
Analysis Server - Kafka Consumer
Kafka에서 텔레메트리 데이터를 읽어 Celery로 추론 작업 제출
"""
import json
import time
import logging
import signal
import sys
import os
from typing import Dict, Any

from confluent_kafka import Consumer, KafkaError, KafkaException

# 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.config.settings import settings
from tasks import celery_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
KAFKA_TOPIC = 'satellite-telemetry'
KAFKA_GROUP_ID = 'analysis-inference-group'

# Global flag for graceful shutdown
running = True


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global running
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    running = False


def create_kafka_consumer() -> Consumer:
    """Create and configure Kafka consumer"""
    conf = {
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        'group.id': KAFKA_GROUP_ID,
        'auto.offset.reset': 'latest',  # 최신 데이터만 처리
        'enable.auto.commit': True,
        'session.timeout.ms': 6000,
        'max.poll.interval.ms': 300000
    }

    consumer = Consumer(conf)
    consumer.subscribe([KAFKA_TOPIC])
    logger.info(f"Kafka consumer created, subscribed to topic: {KAFKA_TOPIC}")
    return consumer


def should_trigger_inference(data: Dict[str, Any]) -> bool:
    """
    추론을 트리거해야 하는지 결정

    조건:
    - 배터리 SoC < 30% (이상 감지)
    - 온도 > 50°C 또는 < -30°C (열 이상)
    - 추진체 작동 중 (궤도 조정)
    - 주기적 추론 (예: 매 10번째 메시지)
    """
    try:
        # EPS 이상
        eps = data.get('eps', {})
        if eps.get('battery_soc_percent', 100) < 30:
            logger.info(f"Low battery detected: {eps.get('battery_soc_percent')}%")
            return True

        # Thermal 이상
        thermal = data.get('thermal', {})
        obc_temp = thermal.get('obc_temp', 0)
        if obc_temp > 50 or obc_temp < -30:
            logger.info(f"Abnormal temperature: {obc_temp}°C")
            return True

        # 추진체 작동
        aocs = data.get('aocs', {})
        if aocs.get('thruster_active', False):
            logger.info("Thruster active - triggering inference")
            return True

        # 주기적 추론 (10% 확률 - 조정 가능)
        import random
        if random.random() < 0.1:
            return True

        return False

    except Exception as e:
        logger.error(f"Error in inference trigger logic: {e}")
        return False


def submit_inference_task(telemetry_data: Dict[str, Any]):
    """
    텔레메트리 데이터를 기반으로 서브시스템별 Celery 추론 작업 제출
    """
    try:
        satellite_id = telemetry_data.get('satellite_id', 'UNKNOWN')
        timestamp = telemetry_data.get('timestamp')
        
        # 트리거 이유 결정
        trigger_reason = determine_trigger_reason(telemetry_data)
        
        # Job ID 생성 (모든 서브시스템 추론을 묶는 단위)
        job_id = f"kafka-{satellite_id}-{int(time.time()*1000)}"
        
        # 서브시스템별 데이터 및 모델 매핑
        subsystems_config = {
            'eps': {
                'model': 'lstm_timeseries',
                'features': ['battery_voltage', 'battery_soc_percent', 'battery_current', 
                            'battery_temperature', 'solar_panel_1_voltage', 'solar_panel_1_current',
                            'solar_panel_2_voltage', 'solar_panel_2_current', 
                            'solar_panel_3_voltage', 'solar_panel_3_current',
                            'total_power_consumption', 'total_power_generation'],
                'forecast_horizon': 5
            },
            'thermal': {
                'model': 'lstm_timeseries',
                'features': ['battery_temp', 'obc_temp', 'comm_temp', 'payload_temp',
                            'solar_panel_temp', 'external_temp'],
                'forecast_horizon': 5
            },
            'aocs': {
                'model': 'lstm_timeseries',
                'features': ['gyro_x', 'gyro_y', 'gyro_z', 'sun_sensor_angle',
                            'magnetometer_x', 'magnetometer_y', 'magnetometer_z',
                            'reaction_wheel_1_rpm', 'reaction_wheel_2_rpm', 'reaction_wheel_3_rpm',
                            'gps_altitude_km', 'gps_velocity_kmps'],
                'forecast_horizon': 3
            },
            'comm': {
                'model': 'lstm_timeseries',
                'features': ['rssi_dbm', 'data_backlog_mb', 'last_contact_seconds_ago'],
                'forecast_horizon': 3
            }
        }
        
        task_ids = []
        
        # 각 서브시스템별로 추론 작업 제출
        for subsystem, config in subsystems_config.items():
            subsystem_data = telemetry_data.get(subsystem, {})
            
            if not subsystem_data:
                continue
            
            # 특징 값 추출
            input_data = []
            input_features = []
            for feature in config['features']:
                value = subsystem_data.get(feature, 0)
                input_data.append(float(value) if value is not None else 0.0)
                input_features.append(feature)
            
            # 데이터가 없으면 스킵
            if not input_data or all(v == 0 for v in input_data):
                continue
            
            # Celery 작업 제출
            task = celery_app.send_task(
                'analysis_server.tasks.run_subsystem_inference',
                args=[
                    job_id,
                    subsystem,
                    config['model'],
                    input_data,
                    input_features,
                    {
                        'forecast_horizon': config['forecast_horizon'],
                        'window_size': 10
                    },
                    {
                        'satellite_id': satellite_id,
                        'source': 'kafka_auto_trigger',
                        'trigger_reason': trigger_reason,
                        'timestamp': timestamp
                    }
                ],
                queue='inference'
            )
            
            task_ids.append(task.id)
            logger.info(f"Submitted {subsystem} inference for {satellite_id}: {task.id}")
        
        if task_ids:
            logger.info(f"Submitted {len(task_ids)} subsystem inferences for {satellite_id} (job: {job_id})")
            return job_id
        else:
            logger.warning(f"No subsystem data available for {satellite_id}")
            return None

    except Exception as e:
        logger.error(f"Error submitting inference tasks: {e}", exc_info=True)
        return None


def determine_trigger_reason(data: Dict[str, Any]) -> str:
    """트리거 이유 결정"""
    eps = data.get('eps', {})
    thermal = data.get('thermal', {})
    aocs = data.get('aocs', {})
    
    reasons = []
    
    if eps.get('battery_soc_percent', 100) < 30:
        reasons.append('low_battery')
    
    obc_temp = thermal.get('obc_temp', 0)
    if obc_temp > 50 or obc_temp < -30:
        reasons.append('abnormal_temperature')
    
    if aocs.get('thruster_active', False):
        reasons.append('thruster_active')
    
    if not reasons:
        reasons.append('periodic_check')
    
    return ','.join(reasons)


def process_telemetry_message(data: Dict[str, Any]):
    """텔레메트리 메시지 처리"""
    try:
        satellite_id = data.get('satellite_id', 'UNKNOWN')

        # 추론 트리거 조건 확인
        if should_trigger_inference(data):
            submit_inference_task(data)
        else:
            logger.debug(f"Skipping inference for {satellite_id} (no trigger condition met)")

    except Exception as e:
        logger.error(f"Error processing telemetry message: {e}", exc_info=True)


def consume_messages():
    """Main consumer loop"""
    consumer = None

    try:
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Create consumer
        consumer = create_kafka_consumer()
        logger.info("Starting inference trigger consumer...")

        message_count = 0

        while running:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.debug(f"Reached end of partition {msg.partition()}")
                elif msg.error():
                    logger.error(f"Consumer error: {msg.error()}")
                    raise KafkaException(msg.error())
                continue

            # Process message
            try:
                telemetry_data = json.loads(msg.value().decode('utf-8'))
                process_telemetry_message(telemetry_data)
                message_count += 1

                if message_count % 100 == 0:
                    logger.info(f"Processed {message_count} telemetry messages")

            except Exception as e:
                logger.error(f"Failed to process message: {e}", exc_info=True)

    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user")
    except Exception as e:
        logger.error(f"Consumer error: {e}", exc_info=True)
    finally:
        if consumer:
            logger.info("Closing consumer...")
            consumer.close()
            logger.info("Consumer closed")


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Analysis Server - Kafka Inference Trigger Consumer")
    logger.info("=" * 60)
    consume_messages()
    logger.info("Consumer shutdown complete")
