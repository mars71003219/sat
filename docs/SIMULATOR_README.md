# 데이터 시뮬레이터 사용 가이드

## 개요

데이터 시뮬레이터는 지속적으로 시계열 데이터를 생성하고 AI 추론 작업을 제출하는 도구입니다.

## 주요 기능

1. **지속적 데이터 생성**: 설정된 주기마다 자동으로 데이터 생성
2. **랜덤 데이터양 조절**: 배치 크기를 동적으로 조절
3. **다양한 패턴 생성**: 5가지 시계열 패턴 지원
   - Linear (선형 증가)
   - Seasonal (계절성)
   - Exponential (지수 증가)
   - Cyclical (순환)
   - Random Walk (랜덤워크)
4. **2개 모델 동시 추론**: LSTM과 Moving Average 모델로 동시 추론
5. **자동 저장**: PostgreSQL, Kafka에 자동 저장

## 시작하기

### 1. 시스템 실행

```bash
docker compose up -d
```

### 2. 기본 실행

```bash
python3 data_simulator.py
```

기본 설정:
- 주기: 5초
- 배치 크기: 1-5개 (랜덤)
- 주기 고정

### 3. 커스텀 설정

**빠른 생성 (3초 주기, 랜덤 주기)**
```bash
python3 data_simulator.py --interval 3 --random-interval
```

**대량 생성 (10-20개 배치)**
```bash
python3 data_simulator.py --min-batch 10 --max-batch 20
```

**느린 생성 (10초 주기, 1개씩)**
```bash
python3 data_simulator.py --interval 10 --min-batch 1 --max-batch 1
```

**모든 옵션 사용**
```bash
python3 data_simulator.py \
    --interval 5 \
    --random-interval \
    --min-batch 2 \
    --max-batch 8 \
    --url http://localhost:8000
```

## 옵션 설명

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--interval` | 데이터 생성 주기 (초) | 5 |
| `--random-interval` | 랜덤 주기 활성화 (interval의 50%-150%) | 비활성화 |
| `--min-batch` | 최소 배치 크기 | 1 |
| `--max-batch` | 최대 배치 크기 | 5 |
| `--url` | Operation Server URL | http://localhost:8000 |

## 웹 대시보드 사용

### 1. 대시보드 열기

브라우저에서 `dashboard.html` 파일을 엽니다:

```bash
# Windows
start dashboard.html

# macOS
open dashboard.html

# Linux
xdg-open dashboard.html
```

또는 직접 브라우저에서 파일을 엽니다.

### 2. 대시보드 기능

대시보드는 실시간으로 다음 정보를 표시합니다:

- **실시간 통계**
  - 총 작업 수
  - 완료된 작업
  - 실패한 작업
  - 성공률
  - 분당 처리량

- **최근 추론 결과**
  - 시간, Job ID, 모델명
  - 패턴 타입, 상태
  - 예측값, 추론 시간

- **연결 상태**
  - WebSocket 연결 상태 표시
  - 자동 재연결 기능

### 3. API 엔드포인트

대시보드는 다음 API를 사용합니다:

- `ws://localhost:8000/api/v1/dashboard/ws` - WebSocket (실시간 업데이트)
- `GET /api/v1/dashboard/recent?limit=20` - 최근 결과 조회
- `GET /api/v1/dashboard/live-stats` - 실시간 통계
- `GET /api/v1/dashboard/model-comparison` - 모델 비교
- `GET /api/v1/dashboard/patterns` - 패턴별 분포

## 데이터 흐름

```
시뮬레이터 생성
    ↓
Operation Server (API)
    ↓
Celery Queue
    ↓
Analysis Worker (추론 실행)
    ↓
결과 저장
    ├─→ PostgreSQL (영구 저장)
    ├─→ Redis (캐시)
    ├─→ Kafka (이벤트 스트림)
    └─→ Elasticsearch (검색)
    ↓
웹 대시보드 (실시간 표시)
```

## 테스트 시나리오

### 시나리오 1: 정상 부하 테스트
```bash
python3 data_simulator.py --interval 5 --min-batch 2 --max-batch 5
```
웹 대시보드에서 실시간 처리 확인

### 시나리오 2: 고부하 테스트
```bash
python3 data_simulator.py --interval 2 --random-interval --min-batch 5 --max-batch 10
```
배치 처리 효율성 확인

### 시나리오 3: 저부하 테스트
```bash
python3 data_simulator.py --interval 10 --min-batch 1 --max-batch 1
```
개별 작업 처리 확인

## 모니터링

### Celery 모니터링 (Flower)
```
http://localhost:5555
```
- Worker 상태 확인
- 작업 큐 모니터링
- 처리 통계

### Kafka 메시지 확인
```bash
docker compose exec kafka kafka-console-consumer \
    --bootstrap-server localhost:9092 \
    --topic inference_results \
    --from-beginning
```

### PostgreSQL 데이터 확인
```bash
docker compose exec postgres psql -U admin -d orders_db
```
```sql
SELECT
    model_name,
    COUNT(*) as total,
    AVG(EXTRACT(EPOCH FROM (completed_at - created_at))) as avg_time
FROM inference_results
WHERE created_at > NOW() - INTERVAL '10 minutes'
GROUP BY model_name;
```

### Redis 캐시 확인
```bash
docker compose exec redis redis-cli
```
```redis
KEYS job:*
GET job:status:<job_id>
```

## 중지

시뮬레이터 중지: `Ctrl + C`

시스템 종료:
```bash
docker compose down
```

## 문제 해결

### 시뮬레이터가 연결할 수 없음
```bash
# Operation Server 상태 확인
docker compose ps operation-server
docker compose logs operation-server --tail 50
```

### 대시보드가 연결 안됨
1. Operation Server가 실행 중인지 확인
2. 브라우저 콘솔에서 에러 확인 (F12)
3. WebSocket 연결 URL 확인 (localhost:8000)

### 추론이 실패함
```bash
# Analysis Worker 로그 확인
docker compose logs analysis-worker-1 --tail 100
```

## 성능 팁

1. **배치 크기 최적화**: GPU 활용을 위해 배치 크기를 4-8로 설정
2. **주기 조절**: 시스템 부하에 따라 interval 조절
3. **Worker 추가**: docker-compose.yml에서 worker 수 증가 가능

## 추가 정보

- 모든 데이터는 UTC 시간으로 저장됩니다
- Job ID는 UUID로 자동 생성됩니다
- 메타데이터에 패턴 정보가 포함되어 분석이 가능합니다
- Kafka 토픽: `inference_results`
