#!/usr/bin/env python3
"""
VictoriaMetrics Kafka Consumer
Consumes satellite telemetry data from Kafka and writes to VictoriaMetrics
"""

import json
import logging
import signal
import sys
import time
from datetime import datetime
from typing import Dict, Any

import requests
from confluent_kafka import Consumer, KafkaError, KafkaException
from dateutil import parser as date_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_BOOTSTRAP_SERVERS = 'kafka:9092'
KAFKA_TOPIC = 'satellite-telemetry'
KAFKA_GROUP_ID = 'victoria-consumer-group'
VICTORIA_METRICS_URL = 'http://victoria-metrics:8428'
VICTORIA_WRITE_ENDPOINT = f'{VICTORIA_METRICS_URL}/api/v1/import/prometheus'

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
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': True,
        'session.timeout.ms': 6000,
        'max.poll.interval.ms': 300000
    }

    consumer = Consumer(conf)
    consumer.subscribe([KAFKA_TOPIC])
    logger.info(f"Kafka consumer created, subscribed to topic: {KAFKA_TOPIC}")
    return consumer


def parse_telemetry_message(msg_value: bytes) -> Dict[str, Any]:
    """Parse Kafka message containing satellite telemetry data"""
    try:
        data = json.loads(msg_value.decode('utf-8'))
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse message as JSON: {e}")
        raise
    except Exception as e:
        logger.error(f"Error parsing message: {e}")
        raise


def format_prometheus_metric(metric_name: str, value: float, labels: Dict[str, str], timestamp_ms: int) -> str:
    """
    Format a single metric in Prometheus exposition format
    
    Example: satellite_temperature{satellite_id="SAT-001"} 23.5 1634308800000
    """
    label_str = ','.join([f'{k}="{v}"' for k, v in labels.items()])
    return f"{metric_name}{{{label_str}}} {value} {timestamp_ms}"


def write_to_victoria_metrics(metrics: list):
    """Write metrics to VictoriaMetrics using Prometheus format"""
    if not metrics:
        return

    # Join all metrics with newlines
    payload = '\n'.join(metrics)

    try:
        response = requests.post(
            VICTORIA_WRITE_ENDPOINT,
            data=payload,
            headers={'Content-Type': 'text/plain'},
            timeout=5
        )

        if response.status_code == 204:
            logger.debug(f"Successfully wrote {len(metrics)} metrics to VictoriaMetrics")
        else:
            logger.warning(f"Unexpected response from VictoriaMetrics: {response.status_code}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error writing to VictoriaMetrics: {e}")


def flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (int, float)):
            items.append((new_key, v))
        elif isinstance(v, bool):
            items.append((new_key, 1 if v else 0))
    return dict(items)


def process_telemetry_data(data: Dict[str, Any]):
    """Convert comprehensive satellite telemetry data to VictoriaMetrics metrics"""
    try:
        # Parse timestamp
        timestamp_str = data.get('timestamp')
        if timestamp_str:
            dt = date_parser.isoparse(timestamp_str)
            timestamp_ms = int(dt.timestamp() * 1000)
        else:
            timestamp_ms = int(time.time() * 1000)

        # Extract satellite ID
        satellite_id = data.get('satellite_id', 'UNKNOWN')

        # Common labels for all metrics
        base_labels = {'satellite_id': satellite_id}

        # Build metrics list
        metrics = []

        # ==========================================
        # Process 6 Major Subsystems
        # ==========================================

        # 1. Beacon & OBC
        beacon = data.get('beacon', {})
        if beacon:
            beacon_flat = flatten_dict(beacon, 'satellite_beacon')
            for key, value in beacon_flat.items():
                if isinstance(value, (int, float)):
                    metrics.append(format_prometheus_metric(key, value, base_labels, timestamp_ms))

        obc = data.get('obc', {})
        if obc:
            obc_flat = flatten_dict(obc, 'satellite_obc')
            for key, value in obc_flat.items():
                if isinstance(value, (int, float)):
                    metrics.append(format_prometheus_metric(key, value, base_labels, timestamp_ms))

        # 2. EPS (전력 시스템) - 핵심 메트릭
        eps = data.get('eps', {})
        if eps:
            # 배터리
            if 'battery_soc_percent' in eps:
                metrics.append(format_prometheus_metric('satellite_battery_soc', eps['battery_soc_percent'], base_labels, timestamp_ms))
            if 'battery_voltage' in eps:
                metrics.append(format_prometheus_metric('satellite_battery_voltage', eps['battery_voltage'], base_labels, timestamp_ms))
            if 'battery_current' in eps:
                metrics.append(format_prometheus_metric('satellite_battery_current', eps['battery_current'], base_labels, timestamp_ms))
            if 'battery_temperature' in eps:
                metrics.append(format_prometheus_metric('satellite_battery_temp', eps['battery_temperature'], base_labels, timestamp_ms))

            # 태양 전지판
            for i in range(1, 4):
                v_key = f'solar_panel_{i}_voltage'
                c_key = f'solar_panel_{i}_current'
                if v_key in eps:
                    metrics.append(format_prometheus_metric(f'satellite_solar_panel_{i}_voltage', eps[v_key], base_labels, timestamp_ms))
                if c_key in eps:
                    metrics.append(format_prometheus_metric(f'satellite_solar_panel_{i}_current', eps[c_key], base_labels, timestamp_ms))

            # 전력 총합
            if 'total_power_consumption' in eps:
                metrics.append(format_prometheus_metric('satellite_power_consumption', eps['total_power_consumption'], base_labels, timestamp_ms))
            if 'total_power_generation' in eps:
                metrics.append(format_prometheus_metric('satellite_power_generation', eps['total_power_generation'], base_labels, timestamp_ms))

        # 3. Thermal (온도)
        thermal = data.get('thermal', {})
        if thermal:
            thermal_mapping = {
                'battery_temp': 'satellite_temp_battery',
                'obc_temp': 'satellite_temp_obc',
                'comm_temp': 'satellite_temp_comm',
                'payload_temp': 'satellite_temp_payload',
                'solar_panel_temp': 'satellite_temp_solar_panel',
                'external_temp': 'satellite_temp_external'
            }
            for key, metric_name in thermal_mapping.items():
                if key in thermal:
                    metrics.append(format_prometheus_metric(metric_name, thermal[key], base_labels, timestamp_ms))

            # 히터/쿨러 상태
            if 'heater_1_on' in thermal:
                metrics.append(format_prometheus_metric('satellite_heater_1_on', 1 if thermal['heater_1_on'] else 0, base_labels, timestamp_ms))
            if 'heater_2_on' in thermal:
                metrics.append(format_prometheus_metric('satellite_heater_2_on', 1 if thermal['heater_2_on'] else 0, base_labels, timestamp_ms))
            if 'cooler_active' in thermal:
                metrics.append(format_prometheus_metric('satellite_cooler_active', 1 if thermal['cooler_active'] else 0, base_labels, timestamp_ms))

        # 4. AOCS (자세 및 궤도)
        aocs = data.get('aocs', {})
        if aocs:
            # 자이로
            if 'gyro_x' in aocs:
                metrics.append(format_prometheus_metric('satellite_gyro_x', aocs['gyro_x'], base_labels, timestamp_ms))
            if 'gyro_y' in aocs:
                metrics.append(format_prometheus_metric('satellite_gyro_y', aocs['gyro_y'], base_labels, timestamp_ms))
            if 'gyro_z' in aocs:
                metrics.append(format_prometheus_metric('satellite_gyro_z', aocs['gyro_z'], base_labels, timestamp_ms))

            # 센서
            if 'sun_sensor_angle' in aocs:
                metrics.append(format_prometheus_metric('satellite_sun_angle', aocs['sun_sensor_angle'], base_labels, timestamp_ms))
            if 'magnetometer_x' in aocs:
                metrics.append(format_prometheus_metric('satellite_mag_x', aocs['magnetometer_x'], base_labels, timestamp_ms))
            if 'magnetometer_y' in aocs:
                metrics.append(format_prometheus_metric('satellite_mag_y', aocs['magnetometer_y'], base_labels, timestamp_ms))
            if 'magnetometer_z' in aocs:
                metrics.append(format_prometheus_metric('satellite_mag_z', aocs['magnetometer_z'], base_labels, timestamp_ms))

            # 리액션 휠
            for i in range(1, 4):
                key = f'reaction_wheel_{i}_rpm'
                if key in aocs:
                    metrics.append(format_prometheus_metric(f'satellite_wheel_{i}_rpm', aocs[key], base_labels, timestamp_ms))

            # 추진체
            if 'thruster_fuel_percent' in aocs:
                metrics.append(format_prometheus_metric('satellite_thruster_fuel', aocs['thruster_fuel_percent'], base_labels, timestamp_ms))
            if 'thruster_pressure_bar' in aocs:
                metrics.append(format_prometheus_metric('satellite_thruster_pressure', aocs['thruster_pressure_bar'], base_labels, timestamp_ms))
            if 'thruster_temperature' in aocs:
                metrics.append(format_prometheus_metric('satellite_thruster_temp', aocs['thruster_temperature'], base_labels, timestamp_ms))
            if 'thruster_active' in aocs:
                metrics.append(format_prometheus_metric('satellite_thruster_active', 1 if aocs['thruster_active'] else 0, base_labels, timestamp_ms))

            # GPS 궤도 정보 (핵심!)
            if 'gps_latitude' in aocs:
                metrics.append(format_prometheus_metric('satellite_latitude', aocs['gps_latitude'], base_labels, timestamp_ms))
            if 'gps_longitude' in aocs:
                metrics.append(format_prometheus_metric('satellite_longitude', aocs['gps_longitude'], base_labels, timestamp_ms))
            if 'gps_altitude_km' in aocs:
                metrics.append(format_prometheus_metric('satellite_altitude', aocs['gps_altitude_km'], base_labels, timestamp_ms))
            if 'gps_velocity_kmps' in aocs:
                metrics.append(format_prometheus_metric('satellite_velocity', aocs['gps_velocity_kmps'], base_labels, timestamp_ms))

        # 5. Comm (통신)
        comm = data.get('comm', {})
        if comm:
            if 'rssi_dbm' in comm:
                metrics.append(format_prometheus_metric('satellite_rssi', comm['rssi_dbm'], base_labels, timestamp_ms))
            if 'tx_active' in comm:
                metrics.append(format_prometheus_metric('satellite_tx_active', 1 if comm['tx_active'] else 0, base_labels, timestamp_ms))
            if 'rx_active' in comm:
                metrics.append(format_prometheus_metric('satellite_rx_active', 1 if comm['rx_active'] else 0, base_labels, timestamp_ms))
            if 'data_backlog_mb' in comm:
                metrics.append(format_prometheus_metric('satellite_data_backlog', comm['data_backlog_mb'], base_labels, timestamp_ms))
            if 'last_contact_seconds_ago' in comm:
                metrics.append(format_prometheus_metric('satellite_last_contact', comm['last_contact_seconds_ago'], base_labels, timestamp_ms))

        # 6. Payload (탑재체)
        payload = data.get('payload', {})
        if payload:
            if 'camera_on' in payload:
                metrics.append(format_prometheus_metric('satellite_camera_on', 1 if payload['camera_on'] else 0, base_labels, timestamp_ms))
            if 'sensor_on' in payload:
                metrics.append(format_prometheus_metric('satellite_sensor_on', 1 if payload['sensor_on'] else 0, base_labels, timestamp_ms))
            if 'payload_temp' in payload:
                metrics.append(format_prometheus_metric('satellite_payload_temp', payload['payload_temp'], base_labels, timestamp_ms))
            if 'payload_power_watts' in payload:
                metrics.append(format_prometheus_metric('satellite_payload_power', payload['payload_power_watts'], base_labels, timestamp_ms))
            if 'images_captured_count' in payload:
                metrics.append(format_prometheus_metric('satellite_images_captured', payload['images_captured_count'], base_labels, timestamp_ms))

        # Write to VictoriaMetrics
        if metrics:
            write_to_victoria_metrics(metrics)
            logger.info(f"Processed telemetry from {satellite_id}: {len(metrics)} metrics")
        else:
            logger.warning(f"No metrics extracted from {satellite_id}")

    except Exception as e:
        logger.error(f"Error processing telemetry data: {e}", exc_info=True)


def consume_messages():
    """Main consumer loop"""
    consumer = None

    try:
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Create consumer
        consumer = create_kafka_consumer()
        logger.info("Starting message consumption...")

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
                telemetry_data = parse_telemetry_message(msg.value())
                process_telemetry_data(telemetry_data)
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
    logger.info("VictoriaMetrics Kafka Consumer Starting...")
    logger.info("=" * 60)
    consume_messages()
    logger.info("Consumer shutdown complete")
