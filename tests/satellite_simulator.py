#!/usr/bin/env python3
"""
인공위성 텔레메트리 데이터 시뮬레이터
실제 위성 센서 데이터와 유사한 패턴을 생성하고 Kafka로 전송합니다.
"""

import json
import math
import random
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, Any
from confluent_kafka import Producer


class SatelliteDataSimulator:
    """
    인공위성 텔레메트리 데이터 생성기

    시뮬레이션되는 센서:
    - 온도 (Temperature): -50°C ~ 50°C, 열 사이클 패턴
    - 고도 (Altitude): 400km ~ 450km, 궤도 변동
    - 속도 (Velocity): 7.6km/s ~ 7.8km/s
    - 배터리 전압 (Battery): 3.0V ~ 4.2V, 충/방전 사이클
    - 태양광 패널 출력 (Solar Power): 0W ~ 100W, 지구 그림자 영향
    """

    def __init__(self, kafka_bootstrap_servers: str = 'localhost:9092',
                 satellite_id: str = 'SAT-001'):
        """
        Args:
            kafka_bootstrap_servers: Kafka 브로커 주소
            satellite_id: 위성 식별자
        """
        self.satellite_id = satellite_id
        self.kafka_topic = 'satellite-telemetry'

        # Kafka Producer 설정
        self.producer = Producer({
            'bootstrap.servers': kafka_bootstrap_servers,
            'client.id': f'satellite-simulator-{satellite_id}'
        })

        # 시뮬레이션 상태 변수
        self.time_elapsed = 0  # 시뮬레이션 경과 시간 (초)
        self.orbit_period = 5400  # 궤도 주기 90분 (5400초)
        self.thermal_cycle_period = 3600  # 열 사이클 60분
        self.battery_cycle_period = 5400  # 배터리 사이클 = 궤도 주기

        # 궤도 위치 (도)
        self.orbital_position = random.uniform(0, 360)

        print(f"Satellite Simulator initialized: {satellite_id}")
        print(f"Kafka: {kafka_bootstrap_servers}")
        print(f"Topic: {self.kafka_topic}")

    def calculate_temperature(self) -> float:
        """
        온도 계산 (-50°C ~ 50°C)
        태양 노출 시 높음, 그림자 진입 시 낮음
        """
        # 기본 온도
        base_temp = 0.0

        # 열 사이클 (태양-그림자 전환)
        thermal_phase = (self.time_elapsed % self.thermal_cycle_period) / self.thermal_cycle_period
        thermal_variation = 30 * math.sin(2 * math.pi * thermal_phase)

        # 태양 노출 여부 (궤도 위치에 따라)
        # 지구 그림자는 궤도의 약 30%
        in_shadow = (self.orbital_position % 360) > 252  # 약 108도 구간
        shadow_factor = -20 if in_shadow else 0

        # 노이즈
        noise = random.gauss(0, 2)

        temperature = base_temp + thermal_variation + shadow_factor + noise
        return round(temperature, 2)

    def calculate_altitude(self) -> float:
        """
        고도 계산 (400km ~ 450km)
        타원 궤도로 인한 변동
        """
        # 평균 고도
        mean_altitude = 425.0

        # 궤도 이심률에 의한 변동
        orbit_phase = (self.time_elapsed % self.orbit_period) / self.orbit_period
        eccentricity_variation = 25 * math.cos(2 * math.pi * orbit_phase)

        # 대기 저항에 의한 점진적 감소 (매우 미미)
        drag_decay = -0.001 * (self.time_elapsed / 3600)  # 시간당 -0.001km

        # 노이즈
        noise = random.gauss(0, 0.5)

        altitude = mean_altitude + eccentricity_variation + drag_decay + noise
        return round(altitude, 2)

    def calculate_velocity(self) -> float:
        """
        속도 계산 (7.6km/s ~ 7.8km/s)
        고도에 반비례 (케플러 법칙)
        """
        # 기본 속도 (평균)
        mean_velocity = 7.66

        # 근지점에서 빠르고, 원지점에서 느림
        orbit_phase = (self.time_elapsed % self.orbit_period) / self.orbit_period
        velocity_variation = 0.1 * math.sin(2 * math.pi * orbit_phase)

        # 노이즈
        noise = random.gauss(0, 0.01)

        velocity = mean_velocity + velocity_variation + noise
        return round(velocity, 3)

    def calculate_battery_voltage(self) -> float:
        """
        배터리 전압 (3.0V ~ 4.2V)
        태양광 충전 및 시스템 소비에 따른 사이클
        """
        # 배터리 충/방전 사이클
        battery_phase = (self.time_elapsed % self.battery_cycle_period) / self.battery_cycle_period

        # 태양광 이용 가능 시 충전, 그림자에서 방전
        in_shadow = (self.orbital_position % 360) > 252

        if in_shadow:
            # 방전 중 (3.0V ~ 3.6V)
            voltage = 3.3 + 0.3 * (1 - battery_phase)
        else:
            # 충전 중 (3.6V ~ 4.2V)
            voltage = 3.6 + 0.6 * battery_phase

        # 노이즈
        noise = random.gauss(0, 0.05)

        voltage = max(3.0, min(4.2, voltage + noise))
        return round(voltage, 2)

    def calculate_solar_power(self) -> float:
        """
        태양광 패널 출력 (0W ~ 100W)
        태양 각도 및 지구 그림자 영향
        """
        # 지구 그림자 확인
        in_shadow = (self.orbital_position % 360) > 252

        if in_shadow:
            # 그림자에서는 출력 없음
            return 0.0

        # 태양 각도에 따른 출력 (최대 100W)
        orbit_phase = (self.time_elapsed % self.orbit_period) / self.orbit_period

        # 코사인 법칙 (태양 각도)
        solar_angle = math.cos(2 * math.pi * orbit_phase)
        max_power = 100.0
        power = max_power * max(0, solar_angle) * 0.85  # 효율 85%

        # 패널 성능 변동
        efficiency_variation = random.uniform(0.95, 1.0)
        power *= efficiency_variation

        # 노이즈
        noise = random.gauss(0, 1)

        power = max(0, power + noise)
        return round(power, 2)

    def calculate_position(self) -> Dict[str, float]:
        """
        위성 위치 계산 (위도, 경도)
        간단한 원형 궤도 시뮬레이션
        """
        # 궤도 각속도
        angular_velocity = 360 / self.orbit_period  # 도/초

        # 경도는 궤도 위치에 따라 변화
        longitude = (self.orbital_position + random.gauss(0, 0.1)) % 360
        if longitude > 180:
            longitude -= 360

        # 위도는 궤도 경사각에 따라 진동 (예: 51.6도 경사)
        inclination = 51.6
        orbit_phase = (self.time_elapsed % self.orbit_period) / self.orbit_period
        latitude = inclination * math.sin(2 * math.pi * orbit_phase)

        latitude += random.gauss(0, 0.1)

        return {
            'latitude': round(latitude, 4),
            'longitude': round(longitude, 4)
        }

    def generate_telemetry(self) -> Dict[str, Any]:
        """텔레메트리 데이터 생성"""
        telemetry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'satellite_id': self.satellite_id,
            'metrics': {
                'temperature': self.calculate_temperature(),
                'altitude': self.calculate_altitude(),
                'velocity': self.calculate_velocity(),
                'battery_voltage': self.calculate_battery_voltage(),
                'solar_power': self.calculate_solar_power()
            },
            'location': self.calculate_position()
        }

        return telemetry

    def delivery_report(self, err, msg):
        """Kafka 메시지 전송 결과 콜백"""
        if err is not None:
            print(f'Message delivery failed: {err}')
        else:
            print(f'Message delivered to {msg.topic()} [{msg.partition()}]')

    def send_telemetry(self, telemetry: Dict[str, Any]):
        """Kafka로 텔레메트리 전송"""
        try:
            # JSON 직렬화
            message = json.dumps(telemetry)

            # Kafka로 전송
            self.producer.produce(
                self.kafka_topic,
                key=self.satellite_id.encode('utf-8'),
                value=message.encode('utf-8'),
                callback=self.delivery_report
            )

            # 즉시 전송 (버퍼링 최소화)
            self.producer.poll(0)

        except Exception as e:
            print(f"Error sending telemetry: {e}")

    def update_orbital_state(self, time_step: float):
        """궤도 상태 업데이트"""
        # 시간 경과
        self.time_elapsed += time_step

        # 궤도 위치 업데이트
        angular_velocity = 360 / self.orbit_period  # 도/초
        self.orbital_position = (self.orbital_position + angular_velocity * time_step) % 360

    def run_simulation(self, interval: float = 5.0, duration: int = None):
        """
        시뮬레이션 실행

        Args:
            interval: 데이터 생성 주기 (초)
            duration: 총 실행 시간 (초), None이면 무한 실행
        """
        print("=" * 70)
        print("인공위성 텔레메트리 시뮬레이터 시작")
        print("=" * 70)
        print(f"위성 ID: {self.satellite_id}")
        print(f"데이터 주기: {interval}초")
        print(f"실행 시간: {'무제한' if duration is None else f'{duration}초'}")
        print("=" * 70)
        print()

        iteration = 0
        start_time = time.time()

        try:
            while True:
                iteration += 1
                current_time = time.time()

                # 종료 조건 확인
                if duration and (current_time - start_time) >= duration:
                    print(f"\n시뮬레이션 완료 (Duration: {duration}초)")
                    break

                # 텔레메트리 생성
                telemetry = self.generate_telemetry()

                # 콘솔 출력
                print(f"[{iteration:04d}] {telemetry['timestamp']}")
                print(f"  Temperature: {telemetry['metrics']['temperature']:6.2f}°C")
                print(f"  Altitude:    {telemetry['metrics']['altitude']:6.2f} km")
                print(f"  Velocity:    {telemetry['metrics']['velocity']:6.3f} km/s")
                print(f"  Battery:     {telemetry['metrics']['battery_voltage']:5.2f} V")
                print(f"  Solar Power: {telemetry['metrics']['solar_power']:6.2f} W")
                print(f"  Position:    ({telemetry['location']['latitude']:7.4f}, "
                      f"{telemetry['location']['longitude']:7.4f})")
                print()

                # Kafka로 전송
                self.send_telemetry(telemetry)

                # 궤도 상태 업데이트
                self.update_orbital_state(interval)

                # 다음 주기까지 대기
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n시뮬레이터 중지")
        finally:
            # Kafka producer 정리
            print(f"총 {iteration}개 메시지 전송")
            print("Producer flushing...")
            self.producer.flush()
            print("Simulator shutdown complete")


def main():
    parser = argparse.ArgumentParser(description='인공위성 텔레메트리 시뮬레이터')

    parser.add_argument(
        '--kafka',
        type=str,
        default='localhost:9092',
        help='Kafka 브로커 주소 (기본값: localhost:9092)'
    )

    parser.add_argument(
        '--satellite-id',
        type=str,
        default='SAT-001',
        help='위성 식별자 (기본값: SAT-001)'
    )

    parser.add_argument(
        '--interval',
        type=float,
        default=5.0,
        help='데이터 생성 주기 (초, 기본값: 5.0)'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='시뮬레이션 실행 시간 (초, 기본값: 무제한)'
    )

    args = parser.parse_args()

    # 시뮬레이터 생성 및 실행
    simulator = SatelliteDataSimulator(
        kafka_bootstrap_servers=args.kafka,
        satellite_id=args.satellite_id
    )

    simulator.run_simulation(
        interval=args.interval,
        duration=args.duration
    )


if __name__ == '__main__':
    main()
