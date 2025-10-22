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
            logger.error(f"Failed to write metrics. Status: {response.status_code}, Response: {response.text}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error writing to VictoriaMetrics: {e}")


def process_telemetry_data(data: Dict[str, Any]):
    """Convert telemetry data to VictoriaMetrics metrics"""
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

        # Extract metrics
        telemetry_metrics = data.get('metrics', {})
        location = data.get('location', {})

        # Build metrics list
        metrics = []

        # Common labels for all metrics
        base_labels = {'satellite_id': satellite_id}

        # Telemetry metrics
        metric_mapping = {
            'temperature': 'satellite_temperature',
            'altitude': 'satellite_altitude',
            'velocity': 'satellite_velocity',
            'battery_voltage': 'satellite_battery_voltage',
            'solar_power': 'satellite_solar_power'
        }

        for key, metric_name in metric_mapping.items():
            if key in telemetry_metrics:
                value = telemetry_metrics[key]
                metric_line = format_prometheus_metric(
                    metric_name,
                    value,
                    base_labels,
                    timestamp_ms
                )
                metrics.append(metric_line)

        # Location metrics (if present)
        if 'latitude' in location and 'longitude' in location:
            metrics.append(format_prometheus_metric(
                'satellite_latitude',
                location['latitude'],
                base_labels,
                timestamp_ms
            ))
            metrics.append(format_prometheus_metric(
                'satellite_longitude',
                location['longitude'],
                base_labels,
                timestamp_ms
            ))

        # Write to VictoriaMetrics
        write_to_victoria_metrics(metrics)

        logger.info(f"Processed telemetry from {satellite_id}: {len(metrics)} metrics")

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

        # Main loop
        while running:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.debug(f"Reached end of partition {msg.partition()}")
                elif msg.error():
                    logger.error(f"Kafka error: {msg.error()}")
                continue

            # Process message
            try:
                telemetry_data = parse_telemetry_message(msg.value())
                process_telemetry_data(telemetry_data)
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                continue

    except KafkaException as e:
        logger.error(f"Kafka exception: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Clean shutdown
        if consumer:
            logger.info("Closing Kafka consumer...")
            consumer.close()
            logger.info("Consumer closed")


def check_victoria_metrics_health():
    """Check if VictoriaMetrics is available"""
    health_url = f"{VICTORIA_METRICS_URL}/health"

    max_retries = 30
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = requests.get(health_url, timeout=2)
            if response.status_code == 200:
                logger.info("VictoriaMetrics is healthy")
                return True
        except requests.exceptions.RequestException:
            pass

        logger.warning(f"VictoriaMetrics not ready, retrying ({attempt + 1}/{max_retries})...")
        time.sleep(retry_delay)

    logger.error("VictoriaMetrics health check failed after max retries")
    return False


def main():
    """Main entry point"""
    logger.info("VictoriaMetrics Kafka Consumer starting...")
    logger.info(f"Kafka: {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"Topic: {KAFKA_TOPIC}")
    logger.info(f"VictoriaMetrics: {VICTORIA_METRICS_URL}")

    # Wait for VictoriaMetrics to be ready
    if not check_victoria_metrics_health():
        logger.error("Cannot connect to VictoriaMetrics, exiting")
        sys.exit(1)

    # Start consuming
    consume_messages()

    logger.info("Consumer shutdown complete")


if __name__ == '__main__':
    main()
