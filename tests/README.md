# 테스트 및 데모 도구

이 디렉토리는 추론 시스템의 테스트와 데모를 위한 도구들을 포함합니다.

##  파일 구조

```
tests/
├── satellite_simulator.py  # 인공위성 텔레메트리 시뮬레이터 (Kafka)
├── data_simulator.py       # 데이터 생성 및 지속적 추론 시뮬레이터
├── demo_system.py          # 시스템 데모
├── test_simulator.py       # 전체 성능 테스트 스위트
├── test_single_model.py    # 단일 모델 빠른 테스트
├── requirements.txt        # Python 의존성
└── README.md               # 이 문서
```

##  주요 테스트 도구

### 1. satellite_simulator.py - 인공위성 텔레메트리 시뮬레이터 (NEW!)

**목적**: 실제 인공위성 센서 데이터를 시뮬레이션하여 VictoriaMetrics 시계열 DB에 저장하고 트렌드 분석을 수행합니다.

**시뮬레이션되는 센서**:
- **온도 (Temperature)**: -50°C ~ 50°C, 열 사이클 및 지구 그림자 영향
- **고도 (Altitude)**: 400km ~ 450km, 타원 궤도 변동
- **속도 (Velocity)**: 7.6km/s ~ 7.8km/s, 케플러 법칙
- **배터리 전압 (Battery)**: 3.0V ~ 4.2V, 충/방전 사이클
- **태양광 출력 (Solar Power)**: 0W ~ 100W, 태양 각도 및 그림자 영향
- **위치 (Location)**: 위도/경도, 궤도 경사각 51.6도

**데이터 흐름**:
```
Satellite Simulator → Kafka (satellite-telemetry) → VictoriaMetrics Consumer → VictoriaMetrics DB
```

**사용법**:
```bash
# 의존성 설치
pip install -r requirements.txt

# 기본 실행 (5초 간격)
python3 satellite_simulator.py

# 커스텀 설정
python3 satellite_simulator.py \
    --kafka kafka:9092 \
    --satellite-id SAT-002 \
    --interval 2.0 \
    --duration 3600

# Docker 외부에서 실행 (로컬 Kafka)
python3 satellite_simulator.py --kafka localhost:9092
```

**파라미터**:
- `--kafka`: Kafka 브로커 주소 (기본값: localhost:9092)
- `--satellite-id`: 위성 식별자 (기본값: SAT-001)
- `--interval`: 데이터 생성 주기 초 (기본값: 5.0)
- `--duration`: 실행 시간 초 (기본값: 무제한)

**출력 예시**:
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

**검증 방법**:
```bash
# VictoriaMetrics에서 데이터 확인
curl "http://localhost:8428/api/v1/query?query=satellite_temperature"

# Kafka 메시지 확인
docker exec -it kafka kafka-console-consumer \
    --bootstrap-server localhost:9092 \
    --topic satellite-telemetry \
    --from-beginning
```

### 2. data_simulator.py - 데이터 시뮬레이터

**목적**: 실제 운영 환경을 시뮬레이션하여 시스템 전체 동작을 검증합니다.

**검증 항목**:
-  VAE & Transformer 모델 추론
-  PostgreSQL 결과 저장
-  Redis 큐 동작
-  RabbitMQ 메시지 처리
-  대시보드 실시간 업데이트
-  Flower Celery 모니터링

**사용법**:
```bash
# 기본 실행 (5초 간격, 1-5개 배치)
python3 data_simulator.py

# 커스텀 설정
python3 data_simulator.py \
    --interval 10 \
    --random-interval \
    --min-batch 2 \
    --max-batch 10

# 다른 서버 테스트
python3 data_simulator.py --url http://remote-server:8000
```

**파라미터**:
- `--interval`: 데이터 생성 주기 (초, 기본값: 5)
- `--random-interval`: 랜덤 주기 활성화 (interval의 50%-150%)
- `--min-batch`: 최소 배치 크기 (기본값: 1)
- `--max-batch`: 최대 배치 크기 (기본값: 5)
- `--url`: Operation Server URL

**생성 패턴**:
- `linear`: 선형 증가 패턴
- `seasonal`: 계절성 패턴
- `random_walk`: 랜덤워크 패턴
- `exponential`: 지수 증가 패턴
- `cyclical`: 순환 패턴

**출력 예시**:
```
======================================================================
데이터 시뮬레이터 시작
======================================================================
기본 주기: 5초
랜덤 주기: 활성화
배치 크기: 1-5개
사용 모델: vae_timeseries, transformer_timeseries
생성 패턴: linear, seasonal, random_walk, exponential, cyclical
======================================================================

[2025-10-15 20:30:00] Iteration #1 - Batch Size: 3
----------------------------------------------------------------------

  [1/3] Pattern: linear, Length: 20
  Data sample: [15.32, 16.89, ..., 35.21]
    - vae_timeseries: Job abc123 submitted
    - transformer_timeseries: Job def456 submitted

  [2/3] Pattern: seasonal, Length: 20
  ...
```

### 2. test_simulator.py - 전체 테스트 스위트

**목적**: 모델의 성능과 정확성을 체계적으로 검증합니다.

**테스트 항목**:
- Health Check
- 단일 요청 테스트 (VAE, Transformer)
- 동시 요청 테스트 (50개, 동시성 10)
- 성능 통계 (Throughput, Latency, Success Rate)

**사용법**:
```bash
python3 test_simulator.py
```

**성능 목표**:
-  Throughput > 30 RPS
-  P95 Latency < 200ms
-  Success Rate > 95%

### 3. test_single_model.py - 단일 모델 테스트

**목적**: 특정 모델의 빠른 동작 확인

**사용법**:
```bash
python3 test_single_model.py
```

**출력**:
```
============================================================
Testing: vae_timeseries
============================================================
Submitting job...
 Job ID: f9e4ea24-2f97-4370-beb0-03551af2e10e
Waiting for result...

 Success!
   Total Time: 1.008s
   Inference Time: 0.102s
   Model Type: VAE
   Predictions: [10.12, 10.27, 9.33, 11.11, 10.90]...
```

### 4. demo_system.py - 시스템 데모

**목적**: 시스템 전체 기능을 데모합니다.

**사용법**:
```bash
python3 demo_system.py
```

##  검증 방법

### 1. PostgreSQL 결과 확인

```bash
docker exec -it postgres psql -U satlas -d satlas_inference

# 최근 결과 조회
SELECT job_id, model_name, status, created_at
FROM inference_results
ORDER BY created_at DESC
LIMIT 10;

# 모델별 통계
SELECT model_name, COUNT(*)
FROM inference_results
WHERE status = 'completed'
GROUP BY model_name;
```

### 2. Redis 큐 확인

```bash
docker exec -it redis redis-cli

# 큐 길이 확인
LLEN celery

# 결과 키 확인
KEYS celery-task-meta-*
```

### 3. 대시보드 확인

브라우저에서 접속:
- **메인 대시보드**: http://localhost
- **실시간 모니터링 대시보드**: http://localhost/dashboard.html

확인 항목:
- 실시간 작업 처리 상태
- 모델별 성능 지표
- 최근 추론 결과
- 성공률 및 처리량

### 4. Flower 모니터링

브라우저에서 접속: http://localhost:5555

확인 항목:
- Celery Worker 상태
- 작업 큐 상태
- 작업 처리 히스토리
- Worker별 성능

### 5. Kafka UI (선택)

브라우저에서 접속: http://localhost:8080

확인 항목:
- Topic 메시지 흐름
- Consumer Group 상태

##  전체 시스템 테스트 워크플로우

### 단계 1: 모든 서비스 시작

```bash
docker compose up -d
```

### 단계 2: 서비스 상태 확인

```bash
docker compose ps
```

모든 서비스가 `Up` 상태여야 합니다:
- triton-server (healthy)
- operation-server
- analysis-worker-1
- postgres
- redis
- kafka
- flower

### 단계 3: 빠른 동작 확인

```bash
cd tests
python3 test_single_model.py
```

 VAE와 Transformer 모두 성공해야 합니다.

### 단계 4: 데이터 시뮬레이터 실행

```bash
python3 data_simulator.py --interval 5 --random-interval --max-batch 3
```

**실행 중 확인**:
1. 터미널에서 작업 제출 로그 확인
2. 대시보드(http://localhost)에서 실시간 업데이트 확인
3. Flower(http://localhost:5555)에서 Worker 활동 확인

**5-10분 실행 후 Ctrl+C로 중지**

### 단계 5: 결과 검증

#### PostgreSQL 확인
```bash
docker exec -it postgres psql -U satlas -d satlas_inference -c \
  "SELECT model_name, COUNT(*) as total,
   AVG(CAST(metrics->>'inference_time' AS FLOAT)) as avg_inference_time
   FROM inference_results
   WHERE status = 'completed'
   GROUP BY model_name;"
```

**기대 결과**:
```
   model_name        | total | avg_inference_time
--------------------+-------+-------------------
 vae_timeseries     |   150 |        0.102
 transformer_timeseries |   150 |        0.104
```

#### Redis 확인
```bash
docker exec -it redis redis-cli INFO stats | grep total_commands_processed
```

#### 대시보드 확인
- 총 작업 수가 증가했는지 확인
- 성공률이 95% 이상인지 확인
- 모델별 분포가 균등한지 확인

### 단계 6: 성능 테스트 (선택)

```bash
python3 test_simulator.py
```

성능 목표 달성 확인:
-  Throughput > 30 RPS
-  P95 Latency < 200ms
-  Success Rate > 95%

##  트러블슈팅

### 문제: 작업이 제출되지만 결과가 나오지 않음

**확인**:
```bash
# Worker 로그 확인
docker compose logs analysis-worker-1 --tail 50

# Triton Server 로그 확인
docker compose logs triton-server --tail 50
```

**해결**: Worker 또는 Triton Server 재시작
```bash
docker compose restart analysis-worker-1 triton-server
```

### 문제: GPU 메모리 부족

**확인**:
```bash
nvidia-smi
```

**해결**: Triton config에서 instance count 줄이기
```
# model_repository/*/config.pbtxt
instance_group [
  {
    count: 1  # 2 → 1로 변경
    kind: KIND_GPU
  }
]
```

### 문제: 데이터베이스 연결 실패

**확인**:
```bash
docker compose logs postgres
docker compose logs operation-server | grep -i database
```

**해결**: PostgreSQL 재시작 및 마이그레이션
```bash
docker compose restart postgres
# operation-server가 자동으로 재연결됩니다
```

##  테스트 체크리스트

시스템 배포 전 다음 항목을 모두 확인하세요:

- [ ] 모든 Docker 컨테이너 정상 실행
- [ ] Triton Server 모델 로드 완료 (VAE, Transformer)
- [ ] test_single_model.py 성공
- [ ] data_simulator.py 5분 실행 성공
- [ ] PostgreSQL에 결과 저장 확인
- [ ] Redis 큐 동작 확인
- [ ] 대시보드 실시간 업데이트 확인
- [ ] Flower에서 Worker 활동 확인
- [ ] GPU 메모리 사용률 < 80%
- [ ] 성공률 > 95%
- [ ] 평균 추론 시간 < 200ms

모든 항목이 체크되면 시스템이 프로덕션 준비 완료입니다! 
