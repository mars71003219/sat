"""
실시간 추론 워커
Kafka에서 위성 텔레메트리 데이터를 소비하고, 슬라이딩 윈도우로 미래 예측을 수행
"""

import os
import json
import time
import logging
from collections import deque
from typing import Dict, List, Deque
from datetime import datetime

import numpy as np
import requests
from confluent_kafka import Consumer, KafkaError

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 설정
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
KAFKA_TOPIC = 'satellite-telemetry'
KAFKA_GROUP_ID = 'inference-worker-group'
TRITON_SERVER_URL = os.getenv('TRITON_SERVER_URL', 'triton-server:8000')
VICTORIA_METRICS_URL = os.getenv('VICTORIA_METRICS_URL', 'http://victoria-metrics:8428')

# 윈도우 설정
WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', '60'))  # 60초 윈도우
PREDICTION_HORIZON = int(os.getenv('PREDICTION_HORIZON', '30'))  # 30초 미래 예측

# 메트릭 목록 (Kafka 메시지의 키 이름)
METRICS = [
    'temperature',
    'altitude',
    'velocity',
    'battery_voltage',
    'solar_power'
]


class InferenceWorker:
    """실시간 추론 워커"""

    def __init__(self):
        """초기화"""
        self.consumer = self._create_consumer()

        # 각 위성별, 메트릭별 슬라이딩 윈도우
        self.windows: Dict[str, Dict[str, Deque]] = {}

        logger.info("Inference Worker 초기화 완료")
        logger.info(f"Kafka: {KAFKA_BOOTSTRAP_SERVERS}")
        logger.info(f"Window Size: {WINDOW_SIZE}s, Prediction Horizon: {PREDICTION_HORIZON}s")

    def _create_consumer(self) -> Consumer:
        """Kafka Consumer 생성"""
        config = {
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
            'group.id': KAFKA_GROUP_ID,
            'auto.offset.reset': 'latest',
            'enable.auto.commit': True
        }
        consumer = Consumer(config)
        consumer.subscribe([KAFKA_TOPIC])
        logger.info(f"Kafka consumer 생성, topic: {KAFKA_TOPIC}")
        return consumer

    def _get_window(self, satellite_id: str, metric: str) -> Deque:
        """윈도우 가져오기 (없으면 생성)"""
        if satellite_id not in self.windows:
            self.windows[satellite_id] = {}

        if metric not in self.windows[satellite_id]:
            self.windows[satellite_id][metric] = deque(maxlen=WINDOW_SIZE)

        return self.windows[satellite_id][metric]

    def _predict(self, satellite_id: str, metric: str, window: List[float]) -> List[float]:
        """
        예측 수행 (Triton 모델이 없으므로 간단한 선형 예측 사용)

        Args:
            satellite_id: 위성 ID
            metric: 메트릭 이름
            window: 입력 윈도우 데이터

        Returns:
            예측 결과 리스트
        """
        try:
            # 간단한 선형 예측 (마지막 10개 값의 트렌드 연장)
            if len(window) < 10:
                # 윈도우가 충분하지 않으면 마지막 값 반복
                predictions = [window[-1]] * PREDICTION_HORIZON
            else:
                # 마지막 10개 값으로 선형 회귀
                recent = window[-10:]
                x = np.arange(len(recent))
                y = np.array(recent)

                # 선형 회귀: y = mx + b
                m = np.cov(x, y)[0, 1] / np.var(x)
                b = np.mean(y) - m * np.mean(x)

                # 미래 예측
                future_x = np.arange(len(recent), len(recent) + PREDICTION_HORIZON)
                predictions = (m * future_x + b).tolist()

            logger.info(f"Prediction for {satellite_id}/{metric}: {len(predictions)} points")
            return predictions

        except Exception as e:
            logger.error(f"예측 실패 ({satellite_id}/{metric}): {e}")
            # 실패 시 마지막 값 반복
            if window:
                return [window[-1]] * PREDICTION_HORIZON
            return []

    def _send_to_victoriametrics(self, satellite_id: str, metric: str,
                                  timestamp: float, predictions: List[float]):
        """
        예측 결과를 VictoriaMetrics에 저장

        현재 시점 + 미래 5초 예측을 모두 저장
        - 과거 예측은 덮어쓰기 되지만, 현재 시점 예측은 계속 축적됨
        - 미래 5초는 항상 최신 예측으로 업데이트됨

        Args:
            satellite_id: 위성 ID
            metric: 메트릭 이름
            timestamp: 현재 타임스탬프
            predictions: 예측값 리스트 (현재~미래 4초, 총 5개)
        """
        try:
            if not predictions:
                return

            # 예측 메트릭 이름 (satellite_ prefix 추가)
            pred_metric = f"satellite_{metric}_prediction"

            # Prometheus 포맷으로 변환 (현재 + 미래 5초)
            lines = []
            for i, pred_value in enumerate(predictions):
                # 현재 타임스탬프 + i초
                pred_timestamp = int((timestamp + i) * 1000)  # milliseconds
                line = f'{pred_metric}{{satellite_id="{satellite_id}"}} {pred_value} {pred_timestamp}'
                lines.append(line)

            # VictoriaMetrics에 전송
            data = '\n'.join(lines)
            response = requests.post(
                f'{VICTORIA_METRICS_URL}/api/v1/import/prometheus',
                data=data,
                headers={'Content-Type': 'text/plain'},
                timeout=5
            )

            if response.status_code == 204:
                logger.info(f"예측 데이터 전송 성공: {satellite_id}/{metric}, {len(predictions)} points")
            else:
                logger.warning(f"VictoriaMetrics 응답 오류: {response.status_code}")

        except Exception as e:
            logger.error(f"VictoriaMetrics 전송 실패: {e}")

    def process_message(self, message: Dict):
        """
        메시지 처리

        Args:
            message: 텔레메트리 메시지
        """
        try:
            satellite_id = message.get('satellite_id')
            timestamp = message.get('timestamp')
            metrics = message.get('metrics', {})

            if not satellite_id or not timestamp:
                logger.warning(f"Invalid message: missing satellite_id or timestamp")
                return

            logger.info(f"Processing message from {satellite_id}, raw timestamp: {timestamp}")

            # 타임스탬프를 초 단위로 변환
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp = dt.timestamp()

            logger.info(f"Converted timestamp: {timestamp} ({datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')})")

            # 각 메트릭 처리
            for metric_name in METRICS:
                if metric_name not in metrics:
                    continue

                value = metrics[metric_name]

                # 윈도우에 추가
                window = self._get_window(satellite_id, metric_name)
                window.append(value)

                # 윈도우 크기 로깅 (처음과 가득 찰 때만)
                if len(window) == 1 or len(window) == WINDOW_SIZE:
                    logger.info(f"Window for {satellite_id}/{metric_name}: {len(window)}/{WINDOW_SIZE}")

                # 윈도우가 가득 찼으면 예측 수행
                if len(window) == WINDOW_SIZE:
                    predictions = self._predict(satellite_id, metric_name, list(window))

                    if predictions:
                        # VictoriaMetrics에 저장
                        self._send_to_victoriametrics(
                            satellite_id,
                            metric_name,
                            timestamp,
                            predictions
                        )

        except Exception as e:
            logger.error(f"메시지 처리 오류: {e}")

    def run(self):
        """워커 실행"""
        logger.info("Inference Worker 시작...")

        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Kafka 오류: {msg.error()}")
                        continue

                # 메시지 파싱
                try:
                    message = json.loads(msg.value().decode('utf-8'))
                    self.process_message(message)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 파싱 오류: {e}")

        except KeyboardInterrupt:
            logger.info("Interrupt received, shutting down...")
        finally:
            self.consumer.close()
            logger.info("Inference Worker 종료")


if __name__ == '__main__':
    worker = InferenceWorker()
    worker.run()
