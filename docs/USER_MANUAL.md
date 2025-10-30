# 위성 텔레메트리 분석 시스템 - 사용자 매뉴얼

## 목차

1. [시스템 시작 및 종료](#1-시스템-시작-및-종료)
2. [메시지 큐 상태 확인](#2-메시지-큐-상태-확인)
3. [데이터베이스 확인](#3-데이터베이스-확인)
4. [시스템 모니터링](#4-시스템-모니터링)
5. [추론 작업 실행](#5-추론-작업-실행)
6. [트러블슈팅](#6-트러블슈팅)
7. [성능 튜닝](#7-성능-튜닝)

---

## 1. 시스템 시작 및 종료

### 1.1 초기 설정

**Kafka 클러스터 ID 생성** (최초 1회만):

```bash
cd /mnt/c/projects/satellite
./init-kafka.sh
```

이 스크립트는 `.env` 파일에 `KAFKA_CLUSTER_ID`를 생성합니다.

### 1.2 전체 시스템 시작

```bash
docker compose up -d
```

**시작 순서**:
1. 인프라 서비스 (Kafka, RabbitMQ, PostgreSQL)
2. AI 서비스 (Triton Server)
3. 애플리케이션 서비스 (Operation Server, Analysis Worker)
4. 웹 서비스 (Frontend, Nginx)

**시작 시간**: 약 30-60초 (Triton Server GPU 초기화 포함)

### 1.3 개별 서비스 시작

```bash
# 특정 서비스만 시작
docker compose up -d <service_name>

# 예시
docker compose up -d operation-server
docker compose up -d analysis-worker-1
```

### 1.4 시스템 상태 확인

```bash
# 모든 서비스 상태
docker compose ps

# 특정 서비스 상태
docker compose ps operation-server
```

**정상 상태 예시**:
```
NAME                   STATUS         PORTS
kafka                  Up 5 minutes   0.0.0.0:9092->9092/tcp
rabbitmq               Up 5 minutes   0.0.0.0:5672->5672/tcp, 0.0.0.0:15672->15672/tcp
postgres               Up 5 minutes   0.0.0.0:5432->5432/tcp
triton-server          Up 5 minutes (healthy)   0.0.0.0:8500-8502->8000-8002/tcp
analysis-worker-1      Up 5 minutes
operation-server       Up 5 minutes   0.0.0.0:8000->8000/tcp
```

### 1.5 시스템 종료

**전체 시스템 종료**:
```bash
docker compose down
```

**데이터 유지하며 종료**:
```bash
docker compose down
# 볼륨은 유지됨 (postgres_data, kafka_data, etc.)
```

**데이터 포함 완전 삭제**:
```bash
docker compose down -v
# 주의: 모든 데이터가 삭제됩니다!
```

### 1.6 로그 확인

```bash
# 전체 로그
docker compose logs

# 특정 서비스 로그
docker compose logs operation-server

# 실시간 로그 (follow)
docker compose logs -f analysis-worker-1

# 최근 N개 라인만
docker compose logs --tail 50 kafka-inference-trigger
```

---

## 2. 메시지 큐 상태 확인

### 2.1 Kafka 상태 확인

#### 2.1.1 Kafka UI (웹 인터페이스)

**접속**: http://localhost:8080

**주요 기능**:
- 토픽 목록 및 상세 정보
- 메시지 실시간 확인
- Consumer Group 상태
- Broker 상태

**확인 사항**:

1. **Topics**:
   ```
   ┌─────────────────────────────────────┐
   │ Topic: satellite-telemetry          │
   │ Partitions: 1                       │
   │ Replication Factor: 1               │
   │ Messages: 1,234                     │
   └─────────────────────────────────────┘
   ```

2. **Consumer Groups**:
   ```
   ┌─────────────────────────────────────┐
   │ Group: analysis-inference-group     │
   │ State: Stable                       │
   │ Members: 1                          │
   │ Lag: 0                              │
   └─────────────────────────────────────┘
   ```

#### 2.1.2 CLI로 Kafka 확인

**토픽 목록**:
```bash
docker exec kafka kafka-topics \
  --bootstrap-server localhost:9092 \
  --list
```

**토픽 상세 정보**:
```bash
docker exec kafka kafka-topics \
  --bootstrap-server localhost:9092 \
  --describe \
  --topic satellite-telemetry
```

**Consumer Group 상태**:
```bash
docker exec kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --describe \
  --group analysis-inference-group
```

**출력 예시**:
```
GROUP                       TOPIC                PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
analysis-inference-group    satellite-telemetry  0          1234           1234            0
```

**중요 지표**:
- `LAG = 0`: 모든 메시지가 처리됨 ✅
- `LAG > 0`: 처리 대기 중인 메시지 존재 ⚠️
- `LAG > 1000`: 처리 지연 발생 ❌

**메시지 소비 확인**:
```bash
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic satellite-telemetry \
  --from-beginning \
  --max-messages 5
```

### 2.2 RabbitMQ 상태 확인

#### 2.2.1 RabbitMQ Management UI

**접속**: http://localhost:15672

**로그인**:
- Username: `guest`
- Password: `guest`

**주요 탭**:

1. **Overview**:
   - 전체 메시지 rate
   - 큐 상태
   - 연결 수

2. **Connections**:
   ```
   ┌─────────────────────────────────────────┐
   │ Client: operation-server                │
   │ State: running                          │
   │ Channels: 1                             │
   └─────────────────────────────────────────┘

   ┌─────────────────────────────────────────┐
   │ Client: analysis-worker-1               │
   │ State: running                          │
   │ Channels: 2                             │
   └─────────────────────────────────────────┘
   ```

3. **Queues**:
   ```
   ┌─────────────────────────────────────────┐
   │ Queue: inference                        │
   │ Ready: 0                                │
   │ Unacked: 2                              │
   │ Total: 2                                │
   │ Incoming: 5.2/s                         │
   │ Deliver / Get: 5.2/s                    │
   └─────────────────────────────────────────┘
   ```

**정상 상태 지표**:
- `Ready: 0~10`: 정상 ✅
- `Ready: > 100`: 처리 지연 ⚠️
- `Unacked`: Worker가 처리 중인 메시지
- `Incoming ≈ Deliver`: 균형 잡힌 처리 ✅

#### 2.2.2 CLI로 RabbitMQ 확인

**큐 목록**:
```bash
docker exec rabbitmq rabbitmqctl list_queues
```

**출력 예시**:
```
Timeout: 60.0 seconds ...
Listing queues for vhost / ...
name            messages
celery          0
inference       2
```

**연결 목록**:
```bash
docker exec rabbitmq rabbitmqctl list_connections
```

**Consumer 목록**:
```bash
docker exec rabbitmq rabbitmqctl list_consumers
```

### 2.3 Celery 작업 모니터링

#### 2.3.1 Flower (웹 인터페이스)

**접속**: http://localhost:5555

**주요 기능**:

1. **Workers** 탭:
   ```
   ┌─────────────────────────────────────────────────┐
   │ Worker: analysis-worker-1                       │
   │ Status: Online                                  │
   │ Active: 2 tasks                                 │
   │ Processed: 1,234 tasks                          │
   │ Failed: 5 tasks                                 │
   │ Succeeded: 1,229 tasks                          │
   │ Retried: 3 tasks                                │
   └─────────────────────────────────────────────────┘
   ```

2. **Tasks** 탭:
   - 실시간 작업 목록
   - 작업 상태 (PENDING, STARTED, SUCCESS, FAILURE)
   - 작업 실행 시간

3. **Monitor** 탭:
   - 실시간 그래프
   - 처리량 (tasks/sec)
   - 성공/실패 비율

**정상 상태 지표**:
- Success Rate: > 95% ✅
- Active Tasks: 0-4 (concurrency=2이므로) ✅
- Failed Tasks: < 5% ⚠️

#### 2.3.2 CLI로 Celery 확인

**Worker 상태**:
```bash
docker exec analysis-worker-1 celery -A tasks inspect active
```

**등록된 Task 목록**:
```bash
docker exec analysis-worker-1 celery -A tasks inspect registered
```

**출력 예시**:
```json
{
  "celery@analysis-worker-1": [
    "analysis_server.tasks.run_inference",
    "analysis_server.tasks.run_subsystem_inference"
  ]
}
```

**Worker 통계**:
```bash
docker exec analysis-worker-1 celery -A tasks inspect stats
```

---

## 3. 데이터베이스 확인

### 3.1 PostgreSQL 접속

#### 3.1.1 psql CLI 접속

```bash
docker exec -it postgres psql -U admin -d orders_db
```

**프롬프트**: `orders_db=#`

#### 3.1.2 주요 쿼리

**테이블 목록**:
```sql
\dt
```

**출력 예시**:
```
                 List of relations
 Schema |          Name           | Type  | Owner
--------+-------------------------+-------+-------
 public | inference_jobs          | table | admin
 public | subsystem_inferences    | table | admin
 public | inference_results       | table | admin
```

**테이블 구조 확인**:
```sql
\d inference_jobs
\d subsystem_inferences
```

### 3.2 추론 결과 조회

#### 3.2.1 최근 작업 조회

```sql
-- 최근 10개 작업
SELECT job_id, satellite_id, source, status, created_at, completed_at
FROM inference_jobs
ORDER BY created_at DESC
LIMIT 10;
```

**출력 예시**:
```
         job_id              | satellite_id |      source        |  status   |       created_at        |      completed_at
-----------------------------+--------------+--------------------+-----------+-------------------------+-------------------------
 COMPLETE-TEST-1761802301    | SAT-001      | manual             | completed | 2025-10-30 12:34:56.789 | 2025-10-30 12:35:01.234
 kafka-SAT-002-1761802245    | SAT-002      | kafka_auto_trigger | completed | 2025-10-30 12:33:00.123 | 2025-10-30 12:33:05.456
```

#### 3.2.2 서브시스템별 결과 조회

```sql
-- 특정 Job의 모든 서브시스템 결과
SELECT job_id, subsystem, model_name, status, anomaly_detected, anomaly_score
FROM subsystem_inferences
WHERE job_id = 'COMPLETE-TEST-1761802301'
ORDER BY subsystem;
```

**출력 예시**:
```
         job_id              | subsystem |   model_name    |  status   | anomaly_detected | anomaly_score
-----------------------------+-----------+-----------------+-----------+------------------+---------------
 COMPLETE-TEST-1761802301    | aocs      | lstm_timeseries | completed | f                |        0.6120
 COMPLETE-TEST-1761802301    | comm      | lstm_timeseries | completed | f                |        0.1970
 COMPLETE-TEST-1761802301    | eps       | lstm_timeseries | completed | f                |        0.2500
 COMPLETE-TEST-1761802301    | thermal   | lstm_timeseries | completed | f                |        0.0480
```

#### 3.2.3 이상 감지 결과 조회

```sql
-- 이상이 감지된 모든 서브시스템
SELECT * FROM v_anomaly_alerts
ORDER BY created_at DESC
LIMIT 10;
```

```sql
-- 특정 위성의 이상 감지 이력
SELECT satellite_id, subsystem, anomaly_score, created_at
FROM v_anomaly_alerts
WHERE satellite_id = 'SAT-001'
ORDER BY created_at DESC;
```

#### 3.2.4 작업 요약 조회

```sql
-- 작업 요약 뷰 사용
SELECT * FROM v_inference_job_summary
WHERE job_id = 'COMPLETE-TEST-1761802301';
```

**출력 예시**:
```
         job_id              | satellite_id | source | trigger_reason | job_status |       created_at        | total_subsystems | completed_subsystems | total_time_seconds | inference_count | anomalies_detected | avg_inference_time_ms |      subsystem_results
-----------------------------+--------------+--------+----------------+------------+-------------------------+------------------+----------------------+--------------------+-----------------+--------------------+-----------------------+----------------------------
 COMPLETE-TEST-1761802301    | SAT-001      | manual | manual_test    | completed  | 2025-10-30 12:34:56.789 |                4 |                    4 |              4.445 |               4 |                  0 |                125.50 | [{"subsystem":"aocs",...}]
```

### 3.3 통계 및 분석 쿼리

#### 3.3.1 전체 통계

```sql
-- 전체 작업 통계
SELECT
    COUNT(*) as total_jobs,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed,
    COUNT(CASE WHEN status = 'processing' THEN 1 END) as processing,
    ROUND(100.0 * COUNT(CASE WHEN status = 'completed' THEN 1 END) / COUNT(*), 2) as success_rate
FROM inference_jobs;
```

**출력 예시**:
```
 total_jobs | completed | failed | processing | success_rate
------------+-----------+--------+------------+--------------
       1250 |      1180 |     65 |          5 |        94.40
```

#### 3.3.2 서브시스템별 통계

```sql
-- 서브시스템별 이상 감지율
SELECT
    subsystem,
    COUNT(*) as total,
    COUNT(CASE WHEN anomaly_detected THEN 1 END) as anomalies,
    ROUND(100.0 * COUNT(CASE WHEN anomaly_detected THEN 1 END) / COUNT(*), 2) as anomaly_rate,
    ROUND(AVG(anomaly_score)::numeric, 4) as avg_score
FROM subsystem_inferences
WHERE status = 'completed'
GROUP BY subsystem
ORDER BY anomaly_rate DESC;
```

**출력 예시**:
```
 subsystem | total | anomalies | anomaly_rate | avg_score
-----------+-------+-----------+--------------+-----------
 eps       |   312 |        15 |         4.81 |    0.3245
 thermal   |   312 |         8 |         2.56 |    0.1890
 aocs      |   312 |         5 |         1.60 |    0.4512
 comm      |   312 |         3 |         0.96 |    0.2134
```

#### 3.3.3 시간대별 처리량

```sql
-- 최근 24시간 시간대별 처리량
SELECT
    DATE_TRUNC('hour', created_at) as hour,
    COUNT(*) as jobs,
    AVG(EXTRACT(EPOCH FROM (completed_at - created_at))) as avg_time_sec
FROM inference_jobs
WHERE created_at > NOW() - INTERVAL '24 hours'
    AND status = 'completed'
GROUP BY DATE_TRUNC('hour', created_at)
ORDER BY hour DESC;
```

#### 3.3.4 모델별 성능

```sql
-- 모델별 평균 추론 시간
SELECT
    model_name,
    COUNT(*) as total,
    ROUND(AVG(inference_time_ms)::numeric, 2) as avg_time_ms,
    ROUND(MIN(inference_time_ms)::numeric, 2) as min_time_ms,
    ROUND(MAX(inference_time_ms)::numeric, 2) as max_time_ms
FROM subsystem_inferences
WHERE status = 'completed'
GROUP BY model_name;
```

### 3.4 데이터 정리

#### 3.4.1 오래된 데이터 삭제

```sql
-- 30일 이상 된 completed 작업 삭제
DELETE FROM inference_jobs
WHERE status = 'completed'
  AND completed_at < NOW() - INTERVAL '30 days';
```

#### 3.4.2 실패한 작업 정리

```sql
-- 실패한 작업만 조회
SELECT job_id, satellite_id, error_message, created_at
FROM inference_jobs
WHERE status = 'failed'
ORDER BY created_at DESC;
```

```sql
-- 7일 이상 된 실패 작업 삭제
DELETE FROM inference_jobs
WHERE status = 'failed'
  AND created_at < NOW() - INTERVAL '7 days';
```

#### 3.4.3 테스트 데이터 삭제

```sql
-- 테스트 작업만 삭제
DELETE FROM inference_jobs
WHERE source = 'test' OR job_id LIKE '%TEST%';
```

### 3.5 데이터베이스 백업

```bash
# 전체 데이터베이스 백업
docker exec postgres pg_dump -U admin orders_db > backup_$(date +%Y%m%d_%H%M%S).sql

# 특정 테이블만 백업
docker exec postgres pg_dump -U admin -t inference_jobs orders_db > inference_jobs_backup.sql
```

### 3.6 데이터베이스 복원

```bash
# 백업 복원
cat backup_20251030_123456.sql | docker exec -i postgres psql -U admin orders_db
```

---

## 4. 시스템 모니터링

### 4.1 VictoriaMetrics 모니터링

**접속**: http://localhost:8428

#### 4.1.1 메트릭 조회 (PromQL)

**브라우저에서**:
```
http://localhost:8428/vmui
```

**주요 메트릭**:

1. **배터리 전압**:
   ```promql
   satellite_battery_voltage{satellite_id="SAT-001"}
   ```

2. **온도**:
   ```promql
   satellite_temperature{satellite_id="SAT-001"}
   ```

3. **전력 소비**:
   ```promql
   satellite_power_consumption{satellite_id="SAT-001"}
   ```

#### 4.1.2 CLI로 메트릭 조회

```bash
# 최신 값 조회
curl 'http://localhost:8428/api/v1/query?query=satellite_temperature{satellite_id="SAT-001"}'

# 시간 범위 조회
curl 'http://localhost:8428/api/v1/query_range?query=satellite_temperature{satellite_id="SAT-001"}&start=2025-10-30T00:00:00Z&end=2025-10-30T23:59:59Z&step=1m'
```

### 4.2 Triton Server 모니터링

#### 4.2.1 Health Check

```bash
# HTTP Health Check
curl http://localhost:8500/v2/health/ready

# 정상 응답: 200 OK (빈 응답 또는 {})
```

#### 4.2.2 모델 상태

```bash
# 모든 모델 목록
curl http://localhost:8500/v2/models

# 특정 모델 상태
curl http://localhost:8500/v2/models/lstm_timeseries/ready
```

**정상 응답**:
```json
{
  "models": [
    {
      "name": "lstm_timeseries",
      "version": "1",
      "state": "READY"
    }
  ]
}
```

#### 4.2.3 Metrics

```bash
# Prometheus 형식 메트릭
curl http://localhost:8502/metrics
```

**주요 메트릭**:
```
# Inference count
nv_inference_count{model="lstm_timeseries",version="1"} 1234

# Inference execution time (microseconds)
nv_inference_exec_time_us{model="lstm_timeseries",version="1"} 125000

# Queue time
nv_inference_queue_time_us{model="lstm_timeseries",version="1"} 5000
```

### 4.3 Docker 리소스 모니터링

```bash
# 모든 컨테이너 리소스 사용량
docker stats

# 특정 컨테이너만
docker stats triton-server analysis-worker-1
```

**출력 예시**:
```
CONTAINER         CPU %   MEM USAGE / LIMIT    MEM %   NET I/O         BLOCK I/O
triton-server     15.2%   2.5GiB / 16GiB       15.6%   1.2MB / 800kB   10MB / 5MB
analysis-worker   5.8%    512MiB / 16GiB       3.2%    500kB / 300kB   2MB / 1MB
```

**주의사항**:
- Triton Server: GPU 메모리는 별도 확인 필요
- CPU > 80%: 과부하 상태 ⚠️
- MEM > 90%: 메모리 부족 위험 ❌

### 4.4 GPU 모니터링 (Triton Server)

```bash
# nvidia-smi 실행
docker exec triton-server nvidia-smi

# 1초마다 업데이트
docker exec triton-server nvidia-smi -l 1
```

**출력 예시**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 13.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA RTX 3090    On   | 00000000:01:00.0 Off |                  N/A |
| 30%   45C    P2    85W / 350W |   1850MiB / 24576MiB |     15%      Default |
+-------------------------------+----------------------+----------------------+
```

**정상 범위**:
- Temperature: < 80°C ✅
- GPU Util: 10-50% (추론 작업 중) ✅
- Memory: < 20GB ✅

---

## 5. 추론 작업 실행

### 5.1 수동 추론 (API)

#### 5.1.1 단일 모델 추론

```bash
curl -X POST http://localhost:8000/api/v1/inference/submit \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "lstm_timeseries",
    "data": [25.0, 26.1, 27.3, 28.5, 29.2, 30.1, 31.0, 32.4, 33.1, 34.5],
    "config": {
      "forecast_horizon": 5,
      "window_size": 10
    },
    "metadata": {
      "satellite_id": "SAT-001",
      "source": "manual"
    }
  }'
```

**응답**:
```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "pending",
  "message": "Inference job submitted to analysis server"
}
```

#### 5.1.2 결과 조회

```bash
# job_id를 변수에 저장
JOB_ID="f47ac10b-58cc-4372-a567-0e02b2c3d479"

# 상태 확인
curl http://localhost:8000/api/v1/inference/status/$JOB_ID

# 결과 조회 (완료 후)
curl http://localhost:8000/api/v1/inference/result/$JOB_ID | jq .
```

### 5.2 자동 추론 (Kafka 트리거)

#### 5.2.1 시뮬레이터 실행

```bash
cd tests
python3 satellite_simulator.py --kafka kafka:9092 --satellites 3 --interval 3 --duration 60
```

**파라미터**:
- `--kafka`: Kafka 주소
- `--satellites`: 위성 수 (1-10)
- `--interval`: 전송 간격 (초)
- `--duration`: 실행 시간 (초)

**출력 예시**:
```
================================================================================
🛰️  인공위성 텔레메트리 시뮬레이터
================================================================================
위성 개수:     3
전송 주기:     3초
실행 시간:     60초
Kafka:        kafka:9092
================================================================================

[SAT-001] Satellite Simulator initialized
[SAT-002] Satellite Simulator initialized
[SAT-003] Satellite Simulator initialized

✅ 3개 위성 초기화 완료

[0001] 12:00:00 | Elapsed: 0s / 60s
[0002] 12:00:03 | Elapsed: 3s / 60s
  [SAT-001] Image captured! (Total: 1)
```

#### 5.2.2 자동 트리거 조건

시스템은 다음 조건에서 자동으로 추론을 실행합니다:

1. **배터리 부족**: `battery_soc_percent < 30%`
2. **온도 이상**: `obc_temp < -30°C` 또는 `> 50°C`
3. **추진제 작동**: `thruster_active = true`
4. **주기적 체크**: 10% 확률

#### 5.2.3 자동 추론 확인

```bash
# Kafka Inference Trigger 로그
docker logs kafka-inference-trigger --tail 50

# Analysis Worker 로그
docker logs analysis-worker-1 --tail 50
```

**정상 로그 예시**:
```
2025-10-30 12:34:56 - Low battery detected: 28%
2025-10-30 12:34:56 - Submitted eps inference for SAT-001: task-id-123
2025-10-30 12:34:56 - Submitted thermal inference for SAT-001: task-id-124
2025-10-30 12:34:56 - Submitted aocs inference for SAT-001: task-id-125
2025-10-30 12:34:56 - Submitted comm inference for SAT-001: task-id-126
2025-10-30 12:34:56 - Submitted 4 subsystem inferences for SAT-001 (job: kafka-SAT-001-1761802496)
```

### 5.3 배치 추론

#### 5.3.1 Python 스크립트

```python
import requests
import time

# 10개 작업 제출
job_ids = []
for i in range(10):
    response = requests.post(
        'http://localhost:8000/api/v1/inference/submit',
        json={
            'model_name': 'lstm_timeseries',
            'data': [25.0 + i * 0.1] * 10,
            'config': {'forecast_horizon': 5}
        }
    )
    job_id = response.json()['job_id']
    job_ids.append(job_id)
    print(f'Submitted job {i+1}/10: {job_id}')
    time.sleep(0.1)

# 결과 대기
print('\nWaiting for results...')
time.sleep(5)

# 결과 조회
for job_id in job_ids:
    result = requests.get(f'http://localhost:8000/api/v1/inference/result/{job_id}')
    if result.status_code == 200:
        print(f'{job_id}: {result.json()["status"]}')
```

---

## 6. 트러블슈팅

### 6.1 일반적인 문제

#### 6.1.1 Kafka 연결 실패

**증상**:
```
Failed to resolve 'kafka:9092': Temporary failure in name resolution
```

**원인**: Kafka가 아직 시작 중이거나 네트워크 문제

**해결**:
```bash
# Kafka 상태 확인
docker compose ps kafka

# Kafka 재시작
docker compose restart kafka

# 30초 대기 후 재시도
```

#### 6.1.2 RabbitMQ 연결 실패

**증상**:
```
[ERROR] Consumer error: Connection refused
```

**해결**:
```bash
# RabbitMQ 상태 확인
docker compose ps rabbitmq

# Health check
docker exec rabbitmq rabbitmq-diagnostics ping

# 재시작
docker compose restart rabbitmq
```

#### 6.1.3 PostgreSQL 연결 실패

**증상**:
```
psycopg2.OperationalError: could not connect to server
```

**해결**:
```bash
# PostgreSQL 상태
docker compose ps postgres

# 로그 확인
docker logs postgres --tail 50

# 재시작
docker compose restart postgres
```

#### 6.1.4 Triton Server 시작 실패

**증상**:
```
triton-server    Exited (1)
```

**원인**: GPU 문제, 모델 로딩 실패

**해결**:
```bash
# 로그 확인
docker logs triton-server

# GPU 확인
nvidia-smi

# 모델 디렉토리 확인
ls -la model_repository/

# 재시작
docker compose restart triton-server
```

### 6.2 성능 문제

#### 6.2.1 Celery Task 지연

**증상**: Flower에서 Task가 PENDING 상태로 대기

**진단**:
```bash
# Worker 상태 확인
docker exec analysis-worker-1 celery -A tasks inspect active

# RabbitMQ 큐 상태
docker exec rabbitmq rabbitmqctl list_queues
```

**해결**:
1. **Worker 수 증가**:
   ```bash
   docker compose up -d --scale analysis-worker-1=3
   ```

2. **Concurrency 증가** (`docker-compose.yml`):
   ```yaml
   command: celery -A tasks worker --loglevel=info --concurrency=4
   ```

3. **Worker 재시작**:
   ```bash
   docker compose restart analysis-worker-1
   ```

#### 6.2.2 Kafka Consumer Lag

**증상**: Consumer lag > 1000

**진단**:
```bash
docker exec kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --describe \
  --group analysis-inference-group
```

**해결**:
1. **Partition 수 증가**:
   ```bash
   docker exec kafka kafka-topics \
     --bootstrap-server localhost:9092 \
     --alter \
     --topic satellite-telemetry \
     --partitions 4
   ```

2. **Consumer 수 증가** (Kafka Inference Trigger):
   ```yaml
   # docker-compose.yml
   kafka-inference-trigger:
     deploy:
       replicas: 2
   ```

### 6.3 데이터 문제

#### 6.3.1 중복 Job ID

**증상**:
```
duplicate key value violates unique constraint "inference_jobs_pkey"
```

**해결**: 이미 수정됨 (ON CONFLICT 사용)

확인:
```sql
SELECT job_id, COUNT(*) FROM inference_jobs GROUP BY job_id HAVING COUNT(*) > 1;
```

#### 6.3.2 JSON 직렬화 오류

**증상**:
```
Object of type bool_ is not JSON serializable
```

**해결**: 이미 수정됨 (numpy 타입 변환)

---

## 7. 성능 튜닝

### 7.1 Celery Worker 최적화

```yaml
# docker-compose.yml
analysis-worker-1:
  command: celery -A tasks worker \
    --loglevel=info \
    --concurrency=4 \
    --max-tasks-per-child=100 \
    --prefetch-multiplier=2
```

**파라미터**:
- `--concurrency`: 동시 처리 작업 수 (CPU 코어 수와 동일 권장)
- `--max-tasks-per-child`: Worker 재시작 주기 (메모리 누수 방지)
- `--prefetch-multiplier`: Prefetch 작업 수 배율

### 7.2 PostgreSQL 최적화

#### 7.2.1 인덱스 확인

```sql
SELECT schemaname, tablename, indexname FROM pg_indexes
WHERE tablename IN ('inference_jobs', 'subsystem_inferences');
```

#### 7.2.2 VACUUM

```sql
-- 테이블 정리 (삭제된 행 정리)
VACUUM ANALYZE inference_jobs;
VACUUM ANALYZE subsystem_inferences;
```

### 7.3 Triton Server 최적화

**Dynamic Batching 설정** (`model_repository/lstm_timeseries/config.pbtxt`):
```protobuf
dynamic_batching {
  preferred_batch_size: [4, 8]
  max_queue_delay_microseconds: 100000
}
```

### 7.4 시스템 모니터링 스크립트

```bash
#!/bin/bash
# monitor.sh - 시스템 상태 요약

echo "=== System Status ==="
echo ""

echo "Services:"
docker compose ps --format "table {{.Service}}\t{{.Status}}"
echo ""

echo "RabbitMQ Queue:"
docker exec rabbitmq rabbitmqctl list_queues | grep inference
echo ""

echo "Kafka Lag:"
docker exec kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --describe \
  --group analysis-inference-group | grep satellite-telemetry
echo ""

echo "PostgreSQL Stats:"
docker exec postgres psql -U admin -d orders_db -c \
  "SELECT COUNT(*) as total,
          COUNT(CASE WHEN status='completed' THEN 1 END) as completed,
          COUNT(CASE WHEN status='processing' THEN 1 END) as processing
   FROM inference_jobs
   WHERE created_at > NOW() - INTERVAL '1 hour';"
echo ""

echo "GPU Status:"
docker exec triton-server nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader
```

**사용**:
```bash
chmod +x monitor.sh
./monitor.sh
```

---

## 부록

### A. 환경 변수

주요 환경 변수 목록 (`.env` 또는 `docker-compose.yml`):

```bash
# Kafka
KAFKA_CLUSTER_ID=<auto-generated>
KAFKA_BOOTSTRAP_SERVERS=kafka:9092

# RabbitMQ
CELERY_BROKER_URL=amqp://guest:guest@rabbitmq:5672//
CELERY_RESULT_BACKEND=rpc://

# PostgreSQL
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin123
POSTGRES_DB=orders_db

# Triton
TRITON_SERVER_URL=triton-server:8001

# VictoriaMetrics
VICTORIA_METRICS_URL=http://victoria-metrics:8428
```

### B. 포트 목록

| 포트 | 서비스 | 설명 |
|------|--------|------|
| 80 | Nginx | 웹 프론트엔드 |
| 8000 | Operation Server | API |
| 8080 | Kafka UI | Kafka 모니터링 |
| 8428 | VictoriaMetrics | 시계열 DB |
| 8500 | Triton HTTP | Triton HTTP API |
| 8501 | Triton gRPC | Triton gRPC API |
| 8502 | Triton Metrics | Triton 메트릭 |
| 9092 | Kafka | 메시지 브로커 |
| 5432 | PostgreSQL | 데이터베이스 |
| 5555 | Flower | Celery 모니터 |
| 5672 | RabbitMQ AMQP | 작업 큐 |
| 15672 | RabbitMQ Management | 큐 관리 UI |
| 9200 | Elasticsearch | 검색 엔진 |

### C. 로그 파일 위치

컨테이너 로그는 Docker에서 관리:
```bash
docker logs <container_name>
```

영구 로그 저장:
```bash
docker logs <container_name> > /path/to/logfile.log 2>&1
```

### D. 유용한 명령어 모음

```bash
# 전체 재시작 (데이터 유지)
docker compose restart

# 특정 서비스만 재빌드
docker compose build --no-cache operation-server
docker compose up -d operation-server

# 로그 실시간 모니터링 (여러 서비스)
docker compose logs -f operation-server analysis-worker-1 kafka-inference-trigger

# 디스크 사용량
docker system df

# 정리 (사용하지 않는 이미지, 컨테이너 삭제)
docker system prune -a

# Volume 목록
docker volume ls

# 특정 Volume 삭제
docker volume rm satellite_postgres_data
```
