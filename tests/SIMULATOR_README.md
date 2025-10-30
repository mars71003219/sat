# ️ 인공위성 텔레메트리 시뮬레이터

실제 위성 데이터 송수신 특성을 모사한 종합 텔레메트리 시뮬레이터입니다.

##  개요

6가지 주요 서브시스템의 실시간 텔레메트리 데이터를 생성합니다:

1. **️ Beacon & OBC** - 위성 생존 신호 및 온보드 컴퓨터
2. ** EPS** - 전력 시스템 (배터리, 태양 전지판)
3. ** Thermal** - 온도 관리 시스템
4. ** AOCS** - 자세 및 궤도 제어 (센서, 추진체, GPS)
5. ** Comm** - 통신 시스템 (RSSI, 데이터 백로그)
6. ** Payload** - 탑재체 (카메라, 센서)

##  실행 방법

### 1. Docker로 실행 (권장)

```bash
docker run -d --rm \
  --name satellite-simulator \
  --network satellite_webnet \
  -v /mnt/c/projects/satellite/tests:/tests \
  -w /tests \
  python:3.10-slim \
  bash -c "pip install -q confluent-kafka && python satellite_simulator.py --kafka kafka:9092 --satellites 10 --interval 5 --duration 5400"
```

### 2. 로컬 실행

```bash
# 의존성 설치
pip install confluent-kafka

# 실행
python satellite_simulator.py --kafka localhost:9092 --satellites 10 --interval 5 --duration 5400
```

##  설정 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--kafka` | `localhost:9092` | Kafka 브로커 주소 |
| `--satellites` | `10` | 위성 개수 |
| `--interval` | `5` | 데이터 전송 주기 (초) |
| `--duration` | `5400` | 실행 시간 (초, 5400=90분) |

### 예시

```bash
# 기본 실행 (10개 위성, 5초 주기, 90분)
python satellite_simulator.py --kafka kafka:9092

# 단일 위성, 1초 주기, 1시간
python satellite_simulator.py --kafka kafka:9092 --satellites 1 --interval 1 --duration 3600

# 많은 위성, 빠른 주기 (부하 테스트)
python satellite_simulator.py --kafka kafka:9092 --satellites 50 --interval 1 --duration 600
```

##  생성되는 텔레메트리 데이터

### Beacon & OBC
```json
{
  "beacon": {
    "alive": true,
    "uptime_seconds": 120,
    "mode": "nominal",
    "last_command_time": "2025-10-30T01:00:00Z"
  },
  "obc": {
    "cpu_usage_percent": 35.2,
    "memory_usage_percent": 48.5,
    "error_count": 0,
    "boot_count": 1
  }
}
```

### EPS (전력 시스템)
```json
{
  "eps": {
    "battery_soc_percent": 85.3,
    "battery_voltage": 4.05,
    "battery_current": 2.3,
    "battery_temperature": 18.5,
    "solar_panel_1_voltage": 7.8,
    "solar_panel_1_current": 2.4,
    "total_power_consumption": 23.5,
    "total_power_generation": 57.2
  }
}
```

### AOCS (자세 및 궤도 제어)
```json
{
  "aocs": {
    "gyro_x": 0.05,
    "gyro_y": -0.02,
    "gyro_z": 0.01,
    "sun_sensor_angle": 45.2,
    "magnetometer_x": 38.5,
    "reaction_wheel_1_rpm": 3020,
    "thruster_active": false,
    "thruster_fuel_percent": 98.5,
    "gps_latitude": 25.5,
    "gps_longitude": 127.3,
    "gps_altitude_km": 435.2,
    "gps_velocity_kmps": 7.66
  }
}
```

### Thermal (온도)
```json
{
  "thermal": {
    "battery_temp": 18.5,
    "obc_temp": 25.3,
    "comm_temp": 22.1,
    "payload_temp": 20.5,
    "external_temp": -15.2,
    "heater_1_on": false,
    "cooler_active": false
  }
}
```

### Comm (통신)
```json
{
  "comm": {
    "rssi_dbm": -78.5,
    "tx_active": true,
    "rx_active": true,
    "data_backlog_mb": 12.5,
    "last_contact_seconds_ago": 0
  }
}
```

### Payload (탑재체)
```json
{
  "payload": {
    "camera_on": true,
    "sensor_on": true,
    "payload_temp": 20.5,
    "payload_power_watts": 25.0,
    "images_captured_count": 15,
    "last_image_time": "2025-10-30T01:15:30Z"
  }
}
```

##  시뮬레이션 이벤트

### 1. Eclipse (지구 그림자)
- 궤도의 약 30% 구간에서 발생
- 태양 전지판 출력 0W
- 배터리 방전 모드
- 온도 하강

### 2. Orbital Maneuver (궤도 조정)
- 30분마다 0.5% 확률로 발생
- 추진체 작동 (30~90초)
- 연료 소모
- 자이로 각속도 증가
- 전력 소비 증가

### 3. Payload Operation (탑재체 작동)
- 배터리 SoC > 40% 조건
- Eclipse 외 구간
- 5% 확률로 이미지 촬영
- 데이터 백로그 증가

### 4. Ground Station Contact (지상국 통신)
- 위도 ±60° 이내에서 가시권
- RSSI 강도 향상
- 데이터 송신 가능
- 백로그 감소

##  모니터링

### Kafka 데이터 확인
```bash
# Kafka 토픽 확인
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092

# 메시지 확인
docker exec -it kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic satellite-telemetry --from-beginning --max-messages 5
```

### 로그 확인
```bash
# Docker 로그
docker logs -f satellite-simulator

# 특정 위성 필터링
docker logs satellite-simulator | grep "SAT-001"
```

##  주요 특징

1. **다중 위성 지원** - 최대 수십~수백 개 위성 동시 시뮬레이션
2. **현실적인 궤도 역학** - LEO 궤도, 90분 주기, eclipse 구간
3. **전력 관리** - 배터리 충/방전, 태양 전지판 각도 영향
4. **열 역학** - Eclipse/태양광 전환 시 온도 변화
5. **자세 제어** - 자이로, 리액션 휠, 추진체 시뮬레이션
6. **통신 가시권** - 위도 기반 지상국 접속 가능 여부
7. **이벤트 기반** - 궤도 조정, 촬영 등 확률적 이벤트

##  위성 모드

| 모드 | 설명 | CPU 사용률 |
|------|------|-----------|
| `safe` | 안전 모드 | ~20% |
| `nominal` | 정상 운용 | ~30% |
| `eclipse` | 지구 그림자 | ~30% |
| `maneuver` | 궤도 조정 | ~50% |
| `payload_active` | 탑재체 작동 | ~70% |

## ️ 문제 해결

### Kafka 연결 실패
```
Failed to resolve 'kafka:9092'
```
**해결**: Docker 네트워크 확인 (`--network satellite_webnet`)

### 메모리 부족
**해결**: 위성 개수 감소 또는 전송 주기 증가