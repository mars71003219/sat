#!/usr/bin/env python3
"""
인공위성 텔레메트리 데이터 시뮬레이터
실제 위성 센서 데이터와 유사한 패턴을 생성하고 Kafka로 전송합니다.
  docker run -d --rm \
    --name satellite-simulator \
    --network satellite_webnet \
    -v /mnt/c/projects/satellite/tests:/tests \
    -w /tests \
    python:3.10-slim \
    bash -c "pip install -q confluent-kafka requests && python satellite_simulator.py --kafka kafka:9092 --interval 1"
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

    추가된 현실적 이벤트:
    - 열 이상 (Thermal anomalies): 간헐적 온도 스파이크
    - 배터리 노화 (Battery degradation): 시간에 따른 성능 저하
    - 궤도 조정 (Orbital maneuvers): 추력기 분사
    - 센서 노이즈 (Sensor drift): 점진적 오차 누적
    - 태양 플레어 영향 (Solar flare impact)
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

        # 이벤트 상태
        self.battery_health = 1.0  # 배터리 건강도 (1.0 = 100%)
        self.thermal_anomaly_active = False
        self.thermal_anomaly_end_time = 0
        self.maneuver_active = False
        self.maneuver_end_time = 0
        self.sensor_drift_temp = 0.0
        self.sensor_drift_altitude = 0.0
        self.last_maneuver_time = -10000  # 마지막 궤도 조정 시간

        print(f"Satellite Simulator initialized: {satellite_id}")
        print(f"Kafka: {kafka_bootstrap_servers}")
        print(f"Topic: {self.kafka_topic}")

    def calculate_temperature(self) -> float:
        """
        온도 계산 (-50°C ~ 50°C)
        태양 노출 시 높음, 그림자 진입 시 낮음
        + 열 이상 이벤트, 센서 드리프트
        """
        # 기본 온도
        base_temp = 0.0

        # 열 사이클 (태양-그림자 전환)
        thermal_phase = (self.time_elapsed % self.thermal_cycle_period) / self.thermal_cycle_period
        thermal_variation = 30 * math.sin(2 * math.pi * thermal_phase)

        # 태양 노출 여부 (궤도 위치에 따라)
        in_shadow = (self.orbital_position % 360) > 252  # 약 108도 구간
        shadow_factor = -20 if in_shadow else 0

        # 그림자 진입/이탈 시 급격한 온도 변화
        shadow_transition_zone = (self.orbital_position % 360)
        if 250 < shadow_transition_zone < 254:  # 그림자 진입
            transition_spike = -15 * (254 - shadow_transition_zone) / 4
        elif 358 < shadow_transition_zone or shadow_transition_zone < 2:  # 그림자 이탈
            if shadow_transition_zone > 180:
                transition_spike = 15 * (360 - shadow_transition_zone) / 2
            else:
                transition_spike = 15 * (2 - shadow_transition_zone) / 2
        else:
            transition_spike = 0

        # 열 이상 이벤트 (간헐적 온도 스파이크) - 5% 확률
        thermal_anomaly = 0
        if not self.thermal_anomaly_active and random.random() < 0.0015:
            self.thermal_anomaly_active = True
            self.thermal_anomaly_end_time = self.time_elapsed + random.uniform(60, 180)
            print(f"  [EVENT] Thermal anomaly detected! (Duration: {self.thermal_anomaly_end_time - self.time_elapsed:.0f}s)")

        if self.thermal_anomaly_active:
            if self.time_elapsed < self.thermal_anomaly_end_time:
                thermal_anomaly = random.uniform(15, 25)
            else:
                self.thermal_anomaly_active = False
                print(f"  [EVENT] Thermal anomaly ended")

        # 센서 드리프트 (점진적 오차 누적)
        self.sensor_drift_temp += random.gauss(0, 0.01)
        self.sensor_drift_temp = max(-2, min(2, self.sensor_drift_temp))  # ±2°C 제한

        # 노이즈
        noise = random.gauss(0, 2)

        temperature = base_temp + thermal_variation + shadow_factor + transition_spike + thermal_anomaly + self.sensor_drift_temp + noise
        return round(temperature, 2)

    def calculate_altitude(self) -> float:
        """
        고도 계산 (400km ~ 450km)
        타원 궤도로 인한 변동 + 궤도 조정 이벤트
        """
        # 평균 고도
        mean_altitude = 425.0

        # 궤도 이심률에 의한 변동
        orbit_phase = (self.time_elapsed % self.orbit_period) / self.orbit_period
        eccentricity_variation = 25 * math.cos(2 * math.pi * orbit_phase)

        # 대기 저항에 의한 점진적 감소
        drag_decay = -0.001 * (self.time_elapsed / 3600)  # 시간당 -0.001km

        # 궤도 조정 (Orbital Maneuver) - 2시간마다 5% 확률
        maneuver_boost = 0
        time_since_last_maneuver = self.time_elapsed - self.last_maneuver_time

        if not self.maneuver_active and time_since_last_maneuver > 7200 and random.random() < 0.002:
            self.maneuver_active = True
            self.maneuver_end_time = self.time_elapsed + random.uniform(30, 90)
            self.last_maneuver_time = self.time_elapsed
            print(f"  [EVENT] Orbital maneuver started! (Burn time: {self.maneuver_end_time - self.time_elapsed:.0f}s)")

        if self.maneuver_active:
            if self.time_elapsed < self.maneuver_end_time:
                # 추력기 분사 중 고도 상승
                burn_progress = (self.time_elapsed - self.last_maneuver_time) / (self.maneuver_end_time - self.last_maneuver_time)
                maneuver_boost = 2.0 * burn_progress  # 최대 2km 상승
            else:
                self.maneuver_active = False
                print(f"  [EVENT] Orbital maneuver completed (+{maneuver_boost:.2f}km)")

        # 센서 드리프트
        self.sensor_drift_altitude += random.gauss(0, 0.005)
        self.sensor_drift_altitude = max(-0.5, min(0.5, self.sensor_drift_altitude))

        # 노이즈
        noise = random.gauss(0, 0.5)

        altitude = mean_altitude + eccentricity_variation + drag_decay + maneuver_boost + self.sensor_drift_altitude + noise
        return round(altitude, 2)

    def calculate_velocity(self) -> float:
        """
        속도 계산 (7.6km/s ~ 7.8km/s)
        고도에 반비례 (케플러 법칙) + 궤도 조정 영향
        """
        # 기본 속도 (평균)
        mean_velocity = 7.66

        # 근지점에서 빠르고, 원지점에서 느림
        orbit_phase = (self.time_elapsed % self.orbit_period) / self.orbit_period
        velocity_variation = 0.1 * math.sin(2 * math.pi * orbit_phase)

        # 궤도 조정 중 속도 변화
        maneuver_delta_v = 0
        if self.maneuver_active:
            # 추력기 분사 중 속도 감소 (고도 상승을 위해)
            maneuver_delta_v = -0.02

        # 노이즈
        noise = random.gauss(0, 0.01)

        velocity = mean_velocity + velocity_variation + maneuver_delta_v + noise
        return round(velocity, 3)

    def calculate_battery_voltage(self) -> float:
        """
        배터리 전압 (3.0V ~ 4.2V)
        태양광 충전 및 시스템 소비에 따른 사이클 + 노화 효과
        """
        # 배터리 노화 (시간에 따라 최대 전압 감소)
        hours_elapsed = self.time_elapsed / 3600
        self.battery_health = max(0.75, 1.0 - (hours_elapsed * 0.00001))  # 매우 느린 노화

        # 배터리 충/방전 사이클
        battery_phase = (self.time_elapsed % self.battery_cycle_period) / self.battery_cycle_period

        # 태양광 이용 가능 시 충전, 그림자에서 방전
        in_shadow = (self.orbital_position % 360) > 252

        if in_shadow:
            # 방전 중 (3.0V ~ 3.6V)
            voltage = 3.3 + 0.3 * (1 - battery_phase)
        else:
            # 충전 중 (3.6V ~ 4.2V)
            max_charge_voltage = 4.2 * self.battery_health
            voltage = 3.6 + (max_charge_voltage - 3.6) * battery_phase

        # 궤도 조정 중 전력 소비 증가
        if self.maneuver_active:
            voltage -= 0.15  # 추력기 작동 시 전압 강하

        # 노이즈
        noise = random.gauss(0, 0.05)

        voltage = max(3.0, min(4.2, voltage + noise))
        return round(voltage, 2)

    def calculate_solar_power(self) -> float:
        """
        태양광 패널 출력 (0W ~ 100W)
        태양 각도 및 지구 그림자 영향 + 패널 성능 저하
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

        # 패널 노화 (시간에 따른 성능 저하)
        hours_elapsed = self.time_elapsed / 3600
        panel_degradation = max(0.85, 1.0 - (hours_elapsed * 0.00002))  # 매우 느린 노화

        power = max_power * max(0, solar_angle) * 0.85 * panel_degradation

        # 패널 성능 변동 (먼지, 각도 미세 조정 등)
        efficiency_variation = random.uniform(0.95, 1.0)
        power *= efficiency_variation

        # 그림자 경계 영역에서 부분 출력
        shadow_position = (self.orbital_position % 360)
        if 248 < shadow_position < 252:  # 그림자 진입 중
            partial_factor = (252 - shadow_position) / 4
            power *= partial_factor
        elif 356 < shadow_position or shadow_position < 4:  # 그림자 이탈 중
            if shadow_position > 180:
                partial_factor = (360 - shadow_position) / 4
            else:
                partial_factor = shadow_position / 4
            power *= partial_factor

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
