# 인공위성 시계열 모니터링 시스템 - 사용 매뉴얼

##  목차

1. [시스템 개요](#시스템-개요)
2. [시작하기](#시작하기)
3. [기본 사용법](#기본-사용법)
4. [고급 기능](#고급-기능)
5. [API 가이드](#api-가이드)
6. [문제 해결](#문제-해결)
7. [FAQ](#faq)

---

## 시스템 개요

### 시스템 목적

인공위성의 텔레메트리 데이터를 실시간으로 수집, 저장, 분석하고 AI 모델을 통해 예측하는 통합 모니터링 시스템입니다.

### 주요 기능

-  **실시간 데이터 수집**: 위성 센서 데이터 시뮬레이션 및 수집
-  **시계열 데이터 저장**: VictoriaMetrics 기반 장기 보관 (1년)
-  **AI 기반 예측**: VAE 및 Transformer 모델을 통한 시계열 예측
-  **트렌드 시각화**: 다크 테마 대시보드로 원본/예측 데이터 비교
-  **실시간 모니터링**: 30초 자동 업데이트
-  **확장 가능한 아키텍처**: 마이크로서비스 기반

### 대상 사용자

-  위성 운영 엔지니어
-  데이터 분석가
-  ML 엔지니어
- ‍ 시스템 관리자

---

## 시작하기

### 시스템 요구사항

#### 하드웨어
- **CPU**: 4코어 이상 권장
- **RAM**: 16GB 이상 권장
- **GPU**: NVIDIA GPU (RTX 5060 이상, 선택사항)
- **디스크**: 50GB 이상 여유 공간

#### 소프트웨어
- **OS**: Linux, macOS, Windows (WSL2)
- **Docker**: 20.10 이상
- **Docker Compose**: 2.0 이상
- **NVIDIA Container Toolkit** (GPU 사용 시)

### 설치 가이드

#### 1. 저장소 클론

```bash
git clone <repository-url>
cd satellite
```

#### 2. 환경 설정

```bash
# Kafka 클러스터 ID 생성
./init-kafka.sh
```

#### 3. 서비스 시작

```bash
# 모든 서비스 시작
docker compose up -d

# 로그 확인
docker compose logs -f
```

#### 4. 서비스 상태 확인

```bash
docker compose ps
```

**예상 출력:**
```
NAME                STATUS
kafka               Up
victoria-metrics    Up
victoria-consumer   Up
operation-server    Up
triton-server       Up (healthy)
analysis-worker-1   Up
postgres            Up
redis               Up
elasticsearch       Up
kafka-ui            Up
flower              Up
web_frontend        Up
web_server          Up
```

### 초기 접속

브라우저에서 다음 URL로 접속:

- **메인 대시보드**: http://localhost
- **API 문서**: http://localhost:8000/docs
- **Kafka UI**: http://localhost:8080
- **Flower (Celery)**: http://localhost:5555
- **RedisInsight**: http://localhost:8001

---

## 기본 사용법

### 1. 위성 데이터 시뮬레이터 실행

#### Docker 컨테이너에서 실행 (권장)

```bash
docker run --rm \
  --network satellite_webnet \
  -v $(pwd)/tests:/tests \
  -w /tests \
  python:3.10-slim \
  bash -c "pip install -q confluent-kafka requests && \
           python satellite_simulator.py \
           --kafka kafka:9092 \
           --interval 5 \
           --satellite-id SAT-001"
```

#### 파라미터 설명

| 파라미터 | 설명 | 기본값 | 예시 |
|----------|------|--------|------|
| `--kafka` | Kafka 브로커 주소 | localhost:9092 | kafka:9092 |
| `--satellite-id` | 위성 식별자 | SAT-001 | SAT-002 |
| `--interval` | 데이터 생성 주기 (초) | 5.0 | 2.0 |
| `--duration` | 실행 시간 (초) | 무제한 | 3600 |

#### 출력 예시

```
======================================================================
인공위성 텔레메트리 시뮬레이터 시작
======================================================================
위성 ID: SAT-001
데이터 주기: 5.0초
실행 시간: 무제한
======================================================================

[0001] 2025-10-22T10:30:00.000000+00:00
  Temperature:  23.45°C
  Altitude:    425.32 km
  Velocity:      7.663 km/s
  Battery:       3.85 V
  Solar Power:  85.23 W
  Position:    (45.2345, 127.5678)

Message delivered to satellite-telemetry [0]
```

### 2. 대시보드 사용

#### 접속

브라우저에서 http://localhost 접속

#### 대시보드 구성 요소

```
┌────────────────────────────────────────────────────────┐
│  [🛰️ Satellite Monitor]  [1h][6h][1d][1w][Custom▼]  │ ← 헤더
├────────────────────────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │
│  │Temp     │ │Altitude │ │Velocity │ │Battery  │     │ ← 메트릭 카드
│  │ 23.5°C  │ │ 425km   │ │ 7.66km/s│ │ 3.85V   │     │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘     │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  Temperature Trend                               │ │
│  │  [━━━━ Raw Data]  [- - - Prediction]           │ │ ← 차트
│  │                                                  │ │
│  │  [차트 영역]                                     │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  Statistics                                      │ │
│  │  Count: 360 | Mean: 23.8 | Min: -5.2 | Max: 48.6│ │ ← 통계
│  └──────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

#### 주요 기능

**1. 메트릭 선택**
- 메트릭 카드 클릭으로 차트 전환
- 5가지 메트릭: Temperature, Altitude, Velocity, Battery, Solar Power

**2. 시간 범위 선택**
- `1h`: 최근 1시간
- `6h`: 최근 6시간 (기본값)
- `1d`: 최근 1일
- `1w`: 최근 1주일

**3. 위성 선택**
- 상단 드롭다운에서 위성 ID 선택
- 여러 위성 데이터 비교 가능

**4. 차트 상호작용**
- **마우스 오버**: 정확한 값 표시
- **파란색 실선**: 원본 센서 데이터
- **녹색 점선**: AI 예측 값

**5. 자동 새로고침**
- 30초마다 자동 업데이트
- 수동 새로고침 버튼 (↻)

### 3. Triton 추론 실행

#### 추론 요청 제출

```bash
curl -X POST "http://localhost:8000/api/v1/inference/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "vae_timeseries",
    "data": [10.5, 11.2, 10.8, 12.1, 11.5, 10.9, 11.8, 12.3,
             11.1, 10.7, 11.4, 12.0, 11.6, 10.4, 11.9, 12.2,
             11.3, 10.6, 11.7, 12.4],
    "config": {
      "sequence_length": 20,
      "forecast_steps": 10
    }
  }'
```

**응답:**
```json
{
  "job_id": "f9e4ea24-2f97-4370-beb0-03551af2e10e",
  "status": "submitted",
  "model_name": "vae_timeseries",
  "submitted_at": "2025-10-22T10:30:00Z"
}
```

#### 결과 조회

```bash
curl "http://localhost:8000/api/v1/inference/result/f9e4ea24-2f97-4370-beb0-03551af2e10e"
```

**응답:**
```json
{
  "job_id": "f9e4ea24-2f97-4370-beb0-03551af2e10e",
  "status": "completed",
  "model_name": "vae_timeseries",
  "result": {
    "predictions": [12.5, 12.1, 11.8, 12.3, 11.9, 12.2, 11.7, 12.0, 11.6, 12.4],
    "mean_prediction": 12.05
  },
  "metrics": {
    "inference_time": 0.102,
    "total_time": 1.234
  }
}
```

### 4. 트렌드 API 사용

#### 원본 데이터 조회

```bash
curl "http://localhost:8000/api/v1/trends/raw?\
metric=satellite_temperature&\
start_time=2025-10-22T00:00:00Z&\
end_time=2025-10-22T12:00:00Z&\
satellite_id=SAT-001"
```

**응답:**
```json
{
  "metric_name": "satellite_temperature",
  "satellite_id": "SAT-001",
  "data_points": [
    {"timestamp": "2025-10-22T00:00:00", "value": 23.5},
    {"timestamp": "2025-10-22T00:01:00", "value": 24.1},
    ...
  ],
  "summary": {
    "count": 720,
    "mean": 23.8,
    "min": -5.2,
    "max": 48.6,
    "std": 12.3
  }
}
```

#### 사용 가능한 메트릭 조회

```bash
curl "http://localhost:8000/api/v1/trends/metrics"
```

**응답:**
```json
{
  "metrics": [
    "satellite_temperature",
    "satellite_altitude",
    "satellite_velocity",
    "satellite_battery_voltage",
    "satellite_solar_power",
    "satellite_latitude",
    "satellite_longitude"
  ],
  "count": 7
}
```

#### 위성 목록 조회

```bash
curl "http://localhost:8000/api/v1/trends/satellites"
```

**응답:**
```json
{
  "satellites": ["SAT-001", "SAT-002"],
  "count": 2
}
```

---

## 고급 기능

### 1. 다중 위성 시뮬레이션

여러 위성의 데이터를 동시에 수집하려면 여러 시뮬레이터를 실행:

**터미널 1: SAT-001**
```bash
docker run --rm --network satellite_webnet \
  -v $(pwd)/tests:/tests -w /tests python:3.10-slim \
  bash -c "pip install -q confluent-kafka && \
           python satellite_simulator.py --kafka kafka:9092 --satellite-id SAT-001"
```

**터미널 2: SAT-002**
```bash
docker run --rm --network satellite_webnet \
  -v $(pwd)/tests:/tests -w /tests python:3.10-slim \
  bash -c "pip install -q confluent-kafka && \
           python satellite_simulator.py --kafka kafka:9092 --satellite-id SAT-002"
```

### 2. VictoriaMetrics 직접 쿼리

#### PromQL을 사용한 고급 쿼리

```bash
# 평균 온도 (최근 1시간)
curl "http://localhost:8428/api/v1/query?query=\
avg_over_time(satellite_temperature[1h])"

# 최대 고도
curl "http://localhost:8428/api/v1/query?query=\
max(satellite_altitude)"

# 배터리 전압 변화율
curl "http://localhost:8428/api/v1/query?query=\
rate(satellite_battery_voltage[5m])"
```

### 3. Kafka 메시지 확인

#### 실시간 메시지 모니터링

```bash
docker exec -it kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic satellite-telemetry \
  --from-beginning
```

#### 메시지 개수 확인

```bash
docker exec kafka kafka-run-class kafka.tools.GetOffsetShell \
  --broker-list localhost:9092 \
  --topic satellite-telemetry
```

### 4. PostgreSQL 데이터 조회

#### 추론 결과 확인

```bash
docker exec -it postgres psql -U admin -d orders_db
```

**SQL 쿼리:**
```sql
-- 최근 추론 결과 10개
SELECT job_id, model_name, status, created_at,
       result->>'mean_prediction' as prediction
FROM inference_results
ORDER BY created_at DESC
LIMIT 10;

-- 모델별 성공률
SELECT model_name,
       COUNT(*) as total,
       SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as success,
       ROUND(100.0 * SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) / COUNT(*), 2) as success_rate
FROM inference_results
GROUP BY model_name;

-- 평균 추론 시간
SELECT model_name,
       AVG(CAST(metrics->>'inference_time' AS FLOAT)) as avg_inference_time,
       MIN(CAST(metrics->>'inference_time' AS FLOAT)) as min_time,
       MAX(CAST(metrics->>'inference_time' AS FLOAT)) as max_time
FROM inference_results
WHERE status = 'completed'
GROUP BY model_name;
```

### 5. Celery 작업 모니터링

#### Flower 대시보드

브라우저에서 http://localhost:5555 접속

**주요 기능:**
- **Workers**: Celery worker 상태 및 통계
- **Tasks**: 실행 중/완료된 작업 목록
- **Monitor**: 실시간 작업 처리 모니터링
- **Broker**: Redis 큐 상태

---

## API 가이드

### REST API 엔드포인트

#### 1. Inference API

**추론 요청 제출**
```
POST /api/v1/inference/submit
Content-Type: application/json

Body:
{
  "model_name": "vae_timeseries" | "transformer_timeseries",
  "data": [float array of length 20],
  "config": {
    "sequence_length": 20,
    "forecast_steps": 10
  },
  "metadata": {} (optional)
}

Response: 202 Accepted
{
  "job_id": "uuid",
  "status": "submitted",
  "model_name": "string",
  "submitted_at": "timestamp"
}
```

**결과 조회**
```
GET /api/v1/inference/result/{job_id}

Response: 200 OK
{
  "job_id": "uuid",
  "status": "completed" | "pending" | "failed",
  "model_name": "string",
  "result": {
    "predictions": [float array],
    "mean_prediction": float
  },
  "metrics": {
    "inference_time": float,
    "total_time": float
  }
}
```

#### 2. Trends API

**원본 데이터 트렌드**
```
GET /api/v1/trends/raw?metric={metric_name}&start_time={iso8601}&end_time={iso8601}&satellite_id={id}

Parameters:
- metric: satellite_temperature, satellite_altitude, etc.
- start_time: ISO 8601 format (required)
- end_time: ISO 8601 format (required)
- satellite_id: SAT-001, etc. (optional)

Response: 200 OK
{
  "metric_name": "string",
  "satellite_id": "string",
  "data_points": [
    {"timestamp": "iso8601", "value": float}
  ],
  "summary": {
    "count": int,
    "mean": float,
    "min": float,
    "max": float,
    "std": float
  }
}
```

**예측 데이터 트렌드**
```
GET /api/v1/trends/prediction?model_name={model}&start_time={iso8601}&end_time={iso8601}&satellite_id={id}

Parameters:
- model_name: vae_timeseries, transformer_timeseries (required)
- start_time, end_time, satellite_id: same as above

Response: Same structure as /trends/raw
```

**비교 분석**
```
GET /api/v1/trends/compare?raw_metric={metric}&model_name={model}&start_time={iso8601}&end_time={iso8601}

Response: 200 OK
{
  "metric_name": "string",
  "raw_data": [...],
  "prediction_data": [...],
  "correlation": float,
  "mae": float,
  "rmse": float
}
```

**메트릭 목록**
```
GET /api/v1/trends/metrics

Response: 200 OK
{
  "metrics": ["satellite_temperature", ...],
  "count": int
}
```

**위성 목록**
```
GET /api/v1/trends/satellites

Response: 200 OK
{
  "satellites": ["SAT-001", ...],
  "count": int
}
```

### API 문서

Swagger UI: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc

---

## 문제 해결

### 일반적인 문제

#### 1. 서비스가 시작되지 않음

**증상:**
```
Error: container exited with code 1
```

**해결:**
```bash
# 로그 확인
docker compose logs [service-name]

# 특정 서비스 재시작
docker compose restart [service-name]

# 전체 재시작
docker compose down && docker compose up -d
```

#### 2. Kafka 연결 실패

**증상:**
```
Connection refused: kafka:9092
```

**해결:**
```bash
# Kafka 상태 확인
docker compose logs kafka

# Kafka 재시작
docker compose restart kafka

# 토픽 확인
docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list
```

#### 3. VictoriaMetrics에 데이터가 없음

**증상:**
- 대시보드에 "No data available"

**해결:**
```bash
# 1. victoria-consumer 로그 확인
docker compose logs victoria-consumer

# 2. VictoriaMetrics 헬스 체크
curl http://localhost:8428/health

# 3. 메트릭 확인
curl "http://localhost:8428/api/v1/label/__name__/values"

# 4. 시뮬레이터 실행 확인
# (시뮬레이터가 실행 중인지 확인)
```

#### 4. GPU 인식 실패

**증상:**
```
Failed to initialize NVML: Driver/library version mismatch
```

**해결:**
```bash
# 1. NVIDIA 드라이버 확인
nvidia-smi

# 2. Docker GPU 런타임 확인
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 3. Triton 재시작
docker compose restart triton-server
```

#### 5. PostgreSQL 연결 실패

**증상:**
```
connection to server failed: Connection refused
```

**해결:**
```bash
# 1. PostgreSQL 로그 확인
docker compose logs postgres

# 2. 연결 테스트
docker exec -it postgres psql -U admin -d orders_db -c "SELECT 1;"

# 3. 재시작
docker compose restart postgres
```

### 로그 확인 방법

```bash
# 모든 서비스 로그
docker compose logs

# 특정 서비스 로그
docker compose logs [service-name]

# 실시간 로그 (tail -f)
docker compose logs -f [service-name]

# 최근 N줄
docker compose logs --tail 100 [service-name]

# 타임스탬프 포함
docker compose logs -t [service-name]
```

### 성능 최적화

#### 1. 메모리 사용량 줄이기

**docker-compose.yml 수정:**
```yaml
services:
  elasticsearch:
    environment:
      - "ES_JAVA_OPTS=-Xms256m -Xmx256m"  # 기본 512m → 256m
```

#### 2. Triton 인스턴스 수 조정

**model_repository/*/config.pbtxt:**
```
instance_group [
  {
    count: 1  # 2 → 1로 변경
    kind: KIND_GPU
  }
]
```

#### 3. Kafka 로그 보존 기간 단축

```bash
docker exec kafka kafka-configs \
  --bootstrap-server localhost:9092 \
  --alter --entity-type topics \
  --entity-name satellite-telemetry \
  --add-config retention.ms=86400000  # 1일
```

---

## FAQ

### Q1: 시스템을 중지하려면?

**A:**
```bash
# 모든 컨테이너 중지 (데이터 유지)
docker compose stop

# 모든 컨테이너 중지 및 제거 (데이터 유지)
docker compose down

# 모든 컨테이너 및 볼륨 제거 (데이터 삭제!)
docker compose down -v
```

### Q2: 데이터는 어디에 저장되나요?

**A:** Docker 볼륨에 저장됩니다:
- `kafka_data`: Kafka 메시지
- `postgres_data`: PostgreSQL 데이터베이스
- `victoria_data`: VictoriaMetrics 시계열 데이터
- `es_data`: Elasticsearch 인덱스

위치: `/var/lib/docker/volumes/satellite_*`

### Q3: 실제 위성 데이터를 사용할 수 있나요?

**A:** 네! `satellite_simulator.py` 대신 실제 데이터 소스를 연결하면 됩니다:

```python
from confluent_kafka import Producer

producer = Producer({'bootstrap.servers': 'kafka:9092'})

# 실제 센서 데이터 읽기
sensor_data = read_from_satellite()  # 사용자 구현

# Kafka로 전송
producer.produce(
    'satellite-telemetry',
    value=json.dumps(sensor_data)
)
```

### Q4: 다른 AI 모델을 추가하려면?

**A:** Triton 모델 저장소에 추가:

1. 모델 디렉토리 생성: `model_repository/my_model/`
2. 모델 파일 추가: `model_repository/my_model/1/model.py`
3. 설정 파일 작성: `model_repository/my_model/config.pbtxt`
4. Triton 재시작: `docker compose restart triton-server`

### Q5: 프로덕션 환경에서 사용할 수 있나요?

**A:** 현재는 개발/테스트용입니다. 프로덕션 사용을 위해서는:

✅ **필수 변경사항:**
- Kafka 클러스터 구성 (3+ 브로커)
- PostgreSQL 복제 설정
- HTTPS/TLS 암호화
- 인증/인가 구현
- 모니터링/알림 강화
- 백업 및 복구 전략

### Q6: 커스텀 메트릭을 추가하려면?

**A:**

1. **시뮬레이터 수정** (`satellite_simulator.py`):
```python
telemetry = {
    'metrics': {
        # 기존 메트릭
        'temperature': ...,
        # 새 메트릭 추가
        'fuel_level': calculate_fuel()
    }
}
```

2. **Consumer 수정** (`victoria-consumer/consumer.py`):
```python
metric_mapping = {
    # 기존 매핑
    'temperature': 'satellite_temperature',
    # 새 매핑 추가
    'fuel_level': 'satellite_fuel_level'
}
```

3. **대시보드 수정** (`frontend/src/components/TrendDashboard.js`):
```javascript
const METRICS = [
  // 기존 메트릭
  { key: 'satellite_temperature', label: 'Temperature', unit: '°C', color: '#4a9eff' },
  // 새 메트릭 추가
  { key: 'satellite_fuel_level', label: 'Fuel', unit: '%', color: '#f59e0b' }
];
```

### Q7: 시스템 리소스 사용량은?

**A:** 평균 사용량 (13개 컨테이너):
- **CPU**: 2-4 코어 (유휴 시)
- **RAM**: ~6GB
- **디스크**: ~10GB (데이터 제외)
- **네트워크**: ~1MB/s (시뮬레이터 실행 시)

### Q8: 백업은 어떻게 하나요?

**A:**

```bash
# 1. VictoriaMetrics 스냅샷
curl http://localhost:8428/snapshot/create

# 2. PostgreSQL 백업
docker exec postgres pg_dump -U admin orders_db > backup.sql

# 3. Kafka 데이터 백업 (선택사항)
docker run --rm -v kafka_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/kafka-backup.tar.gz /data
```

### Q9: 개발 환경과 프로덕션 환경의 차이는?

**A:**

| 항목 | 개발 (현재) | 프로덕션 (권장) |
|------|-------------|-----------------|
| Kafka | 1 broker | 3+ brokers |
| Replication | 1 | 3 |
| PostgreSQL | Single instance | Master-Slave |
| VictoriaMetrics | Single node | Cluster |
| TLS/SSL | 없음 | 필수 |
| 인증 | 없음 | 필수 |
| 모니터링 | Basic | Prometheus + Grafana |
| 로그 | Docker logs | ELK Stack |

### Q10: 시스템을 업데이트하려면?

**A:**

```bash
# 1. 최신 코드 가져오기
git pull

# 2. 이미지 빌드
docker compose build

# 3. 서비스 재시작 (무중단)
docker compose up -d

# 4. 특정 서비스만 업데이트
docker compose up -d --no-deps --build [service-name]
```

---

## 추가 리소스

### 문서
- 시스템 구성도: `/docs/SYSTEM_ARCHITECTURE.md`
- UML 다이어그램: `/docs/UML_DIAGRAMS.md`
- Kafka 아키텍처: `/docs/KAFKA_ARCHITECTURE.md`
- 위성 트렌드 시스템: `/docs/SATELLITE_TREND_SYSTEM.md`

### 지원
- GitHub Issues: <repository-url>/issues
- 이메일: support@example.com

### 라이선스
MIT License
